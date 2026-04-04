import { DocumentData, LineItem, ReportData, ComparisonResult, ComparisonDetail, MatchStatus, AIProvider, CompareField } from '../types';
import { getEmbeddings } from './gemini';
import { getEmbeddingsOpenAI } from './openai';

const DEFAULT_COMPARE_FIELDS: CompareField[] = ['itemName', 'itemCode', 'unit', 'quantity', 'unitPrice', 'totalPrice'];

// Cosine similarity for semantic matching
function cosineSimilarity(vecA: number[], vecB: number[]): number {
  let dotProduct = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < vecA.length; i++) {
    dotProduct += vecA[i] * vecB[i];
    normA += vecA[i] * vecA[i];
    normB += vecB[i] * vecB[i];
  }
  if (normA === 0 || normB === 0) return 0;
  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

function normalizeText(value: string | null | undefined): string {
  return String(value || '').trim().replace(/\s+/g, ' ').toLowerCase();
}

function stripVietnameseAccents(value: string): string {
  return value.normalize('NFD').replace(/[\u0300-\u036f]/g, '');
}

function normalizeUnitSmart(value: string | null | undefined): string {
  const raw = String(value || '');
  let text = stripVietnameseAccents(raw).toLowerCase().trim();
  text = text.replace(/[\s\.]+/g, ' ');
  text = text.replace(/[^a-z0-9 ]+/g, '');
  const normalized = text.replace(/\s+/g, ' ').trim();
  const unitMap: Record<string, string> = {
    cai: 'cai', cái: 'cai', caí: 'cai',
    bo: 'bo', bộ: 'bo',
    lo: 'lo', lọ: 'lo',
    hop: 'hop', hộp: 'hop',
    vien: 'vien', viên: 'vien',
    tap: 'tap', tập: 'tap',
    tui: 'tui', túi: 'tui',
    vi: 'vi', vỉ: 'vi'
  };
  return unitMap[normalized] ?? normalized;
}

function isHardEquivalentUnit(a: string | null | undefined, b: string | null | undefined): boolean {
  const normA = normalizeUnitSmart(a);
  const normB = normalizeUnitSmart(b);
  if (!normA && !normB) return true;
  if (!normA || !normB) return false;
  return normA === normB;
}

function isSoftEquivalentUnit(a: string | null | undefined, b: string | null | undefined, baseItem: LineItem, matchedItem: LineItem): boolean {
  if (isHardEquivalentUnit(a, b)) return true;
  const normA = normalizeUnitSmart(a);
  const normB = normalizeUnitSmart(b);
  if (normA === normB) return true;

  const codeCompatible = areProductCodesCompatible(baseItem, matchedItem);
  const nameSimilarity = calculateNameSimilarity(baseItem.itemName, matchedItem.itemName);
  const strongName = nameSimilarity >= 0.88;
  const stickyContext = isStickyNoteProduct(baseItem.itemName) || isStickyNoteProduct(matchedItem.itemName);

  if (!codeCompatible || !strongName) return false;
  if (stickyContext && ((normA === 'tap' && normB === 'cai') || (normA === 'cai' && normB === 'tap'))) {
    return true;
  }

  return false;
}

function shouldIgnoreUnitDifference(baseItem: LineItem, matchedItem: LineItem): boolean {
  return isSoftEquivalentUnit(baseItem.unit, matchedItem.unit, baseItem, matchedItem);
}

function normalizeProductNameSmart(value: string | null | undefined): string {
  let text = String(value || '');
  text = stripVietnameseAccents(text).toLowerCase();
  text = text.replace(/[\[\]\(\)\{\}]/g, ' ');
  text = text.replace(/[\u2010-\u2015]/g, '-');
  text = text.replace(/[_\/,:;]+/g, ' ');
  text = text.replace(/(?:^|\s)mg(?:\s|$)/g, ' ');
  text = text.replace(/-+\s*mg\b/g, ' ');
  text = text.replace(/\bmg\b/g, ' ');
  text = text.replace(/(\d)\s*[,\.]\s*(\d)/g, '$1.$2');
  text = text.replace(/(\d)\s*cm\b/g, '$1cm');
  text = text.replace(/([a-zA-Z]+)(\d+mm)\b/g, '$1 $2');
  text = text.replace(/[-–—_]+/g, ' ');
  text = text.replace(/\s+/g, ' ');
  return text.trim();
}

function normalizeMatchText(value: string | null | undefined): string {
  return String(value || '')
    .toLowerCase()
    .replace(/[_-]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function stripLeadingCode(value: string): string {
  return value.replace(/^[a-z]{0,5}\d+[a-z0-9]*\s*/i, '').trim();
}

function levenshteinDistance(a: string, b: string): number {
  const rows = a.length + 1;
  const cols = b.length + 1;
  const dp: number[][] = Array.from({ length: rows }, () => Array(cols).fill(0));

  for (let i = 0; i < rows; i++) dp[i][0] = i;
  for (let j = 0; j < cols; j++) dp[0][j] = j;

  for (let i = 1; i < rows; i++) {
    for (let j = 1; j < cols; j++) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      dp[i][j] = Math.min(
        dp[i - 1][j] + 1,
        dp[i][j - 1] + 1,
        dp[i - 1][j - 1] + cost
      );
    }
  }

  return dp[a.length][b.length];
}

function stripProductCodeFromName(value: string): string {
  const productCode = extractProductCode(value);
  if (!productCode) return value;
  const escaped = productCode.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  return value.replace(new RegExp(`\\b${escaped}\\b`, 'gi'), ' ').replace(/\s+/g, ' ').trim();
}

function extractAllProductCodes(text: string | null | undefined): string[] {
  const normalized = String(text || '').toUpperCase();
  const codePatterns = [/[A-Z]{2,}-\d{1,}/g, /[A-Z]{2,}\d+[A-Z0-9]*/g, /\d{3,}/g];
  const codes: string[] = [];
  for (const pattern of codePatterns) {
    const matches = normalized.match(pattern) || [];
    for (const match of matches) {
      const clean = normalizeCodeText(match);
      if (!clean) continue;
      if (!codes.includes(clean)) {
        codes.push(clean);
      }
    }
  }
  return codes;
}

function extractProductCode(text: string | null | undefined): string | null {
  return extractAllProductCodes(text)[0] || null;
}

function normalizeCodeText(value: string | null | undefined): string {
  return String(value || '')
    .toUpperCase()
    .replace(/[\s_-]+/g, '')
    .trim();
}

function isLikelyInternalNumericCode(value: string | null | undefined): boolean {
  const normalized = normalizeCodeText(value);
  return /^\d{3,}$/.test(normalized);
}

function isLikelyProductCode(value: string | null | undefined): boolean {
  const normalized = normalizeCodeText(value);
  return normalized.length >= 4 && /[A-Z]/.test(normalized) && /\d/.test(normalized);
}

function getAllItemCodes(item: LineItem): string[] {
  const seen = new Set<string>();
  const itemCode = normalizeCodeText(item.itemCode);
  if (itemCode) {
    seen.add(itemCode);
  }
  for (const code of extractAllProductCodes(item.itemName)) {
    seen.add(normalizeCodeText(code));
  }
  return Array.from(seen);
}

function getAliasProductCodes(item: LineItem): string[] {
  const primary = getPrimaryProductCode(item);
  return getAllItemCodes(item).filter(code => primary ? normalizeCodeText(code) !== normalizeCodeText(primary) : true);
}

function isLikelyPrimaryProductCode(code: string): boolean {
  const normalized = normalizeCodeText(code);
  if (isLikelyInternalNumericCode(normalized)) return false;
  if (/^[A-Z]{2,}-\d{1,}$/.test(normalized)) return true;
  if (/^[A-Z]+\d+[A-Z0-9]*$/.test(normalized) && normalized.length <= 12) return true;
  return false;
}

function getPrimaryProductCode(item: LineItem): string | null {
  const itemCode = normalizeCodeText(item.itemCode);
  if (itemCode && isLikelyProductCode(itemCode)) {
    return itemCode;
  }

  const codes = extractAllProductCodes(item.itemName);
  if (codes.length === 0) {
    return itemCode || null;
  }

  const scored = codes.map((code, index) => {
    const normalized = normalizeCodeText(code);
    let score = 100 - index * 5;
    if (/^[A-Z]{2,}-\d{1,}$/.test(normalized)) score += 20;
    if (/^[A-Z]+\d+[A-Z0-9]*$/.test(normalized)) score += 10;
    if (normalized.length <= 8) score += 8;
    if (normalized.length >= 12) score -= 5;
    if (itemCode && normalized === itemCode) score += 25;
    if (isLikelyInternalNumericCode(normalized)) score -= 30;
    return { normalized, score };
  });

  scored.sort((a, b) => b.score - a.score);
  return scored[0]?.normalized || itemCode || null;
}

function areProductCodesCompatible(baseItem: LineItem, matchedItem: LineItem): boolean {
  const basePrimary = getPrimaryProductCode(baseItem);
  const matchedPrimary = getPrimaryProductCode(matchedItem);
  const baseCodes = new Set(getAllItemCodes(baseItem));
  const matchedCodes = new Set(getAllItemCodes(matchedItem));

  if (basePrimary && matchedPrimary && normalizeCodeText(basePrimary) === normalizeCodeText(matchedPrimary)) {
    return true;
  }

  if (basePrimary && matchedCodes.has(normalizeCodeText(basePrimary))) {
    return true;
  }
  if (matchedPrimary && baseCodes.has(normalizeCodeText(matchedPrimary))) {
    return true;
  }

  const commonCodes = [...baseCodes].filter(code => matchedCodes.has(code));
  const commonProductCodes = commonCodes.filter(code => !isLikelyInternalNumericCode(code));
  if (commonProductCodes.length > 0) {
    const nameSimilarity = calculateNameSimilarity(baseItem.itemName, matchedItem.itemName);
    return nameSimilarity >= 0.86;
  }

  return false;
}

function isStickyNoteProduct(value: string | null | undefined): boolean {
  const normalized = normalizeProductNameSmart(value);
  return /giay nho|sticky|memo note|memo notes|note pad|giay ghi chu|block giay|block giay nho|giay note|giay ghi chu/.test(normalized);
}

function calculateNameSimilarity(a: string, b: string): number {
  const rawA = normalizeProductNameSmart(a);
  const rawB = normalizeProductNameSmart(b);

  if (!rawA || !rawB) return 0;

  // Pass 1: Exact phrase match after full normalization → perfect score
  if (rawA === rawB) return 1;

  const cleanA = stripProductCodeFromName(rawA);
  const cleanB = stripProductCodeFromName(rawB);

  if (cleanA === cleanB) return 0.98;
  if (cleanA.includes(cleanB) || cleanB.includes(cleanA)) return 0.94;
  if (stripLeadingCode(cleanA) === stripLeadingCode(cleanB)) return 0.92;

  // Pass 2: Fuzzy edit-distance score (semantic vectors handled by caller)
  const dist = levenshteinDistance(cleanA, cleanB);
  const maxLen = Math.max(cleanA.length, cleanB.length);
  const score = maxLen > 0 ? 1 - dist / maxLen : 0;

  return score;
}

function calculateProductCodeSimilarity(a: string | null | undefined, b: string | null | undefined): number {
  const codeA = normalizeCodeText(a);
  const codeB = normalizeCodeText(b);

  if (!codeA || !codeB) return 0;
  if (codeA === codeB) return 1;
  if (codeA.includes(codeB) || codeB.includes(codeA)) return 0.93;

  const digitsA = (codeA.match(/\d+/g) || []).join('');
  const digitsB = (codeB.match(/\d+/g) || []).join('');
  if (digitsA && digitsB && digitsA === digitsB) return 0.85;

  return 0;
}

function calculateCodeSimilarity(a: string | null | undefined, b: string | null | undefined): number {
  const codeA = normalizeCodeText(a);
  const codeB = normalizeCodeText(b);

  if (!codeA || !codeB) return 0;
  if (codeA === codeB) return 1;
  if (codeA.includes(codeB) || codeB.includes(codeA)) return 0.93;

  const digitsA = (codeA.match(/\d+/g) || []).join('');
  const digitsB = (codeB.match(/\d+/g) || []).join('');
  if (digitsA && digitsB && digitsA === digitsB) return 0.9;

  return 0;
}

// ─────────────────────────────────────────────────────────────────────────────

function isFieldDifferent(field: CompareField, baseItem: LineItem, matchedItem: LineItem): boolean {
  switch (field) {
    case 'itemName':
      return normalizeText(baseItem.itemName) !== normalizeText(matchedItem.itemName);
    case 'itemCode':
      return normalizeText(baseItem.itemCode) !== normalizeText(matchedItem.itemCode);
    case 'unit':
      return !shouldIgnoreUnitDifference(baseItem, matchedItem);
    case 'quantity':
      return baseItem.quantity !== matchedItem.quantity;
    case 'unitPrice':
      return baseItem.unitPrice !== matchedItem.unitPrice;
    case 'totalPrice':
      return baseItem.totalPrice !== matchedItem.totalPrice;
    default:
      return false;
  }
}

export async function generateReport(
  rawDocuments: DocumentData[],
  baseFileName?: string | null,
  aiProvider: AIProvider = 'openai',
  compareFields: CompareField[] = DEFAULT_COMPARE_FIELDS,
): Promise<ReportData> {
  if (rawDocuments.length === 0) throw new Error('No documents provided');

  const activeCompareFields = compareFields.length > 0 ? compareFields : DEFAULT_COMPARE_FIELDS;

  // Filter out non-products and aggregate identical items (same name & code)
  const documents = rawDocuments.map(doc => {
    const validItems = doc.lineItems.filter(item => {
      const hasSomeNumber = item.quantity !== null || item.unitPrice !== null || item.totalPrice !== null;
      const isExcluded = /^(tổng|cộng|chiết khấu|thuế|vat|tiền hàng|giảm giá|phí|thanh toán)/i.test(item.itemName.trim());
      return hasSomeNumber && !isExcluded;
    });

    const groupedItems: LineItem[] = [];

    for (const item of validItems) {
      const codeKey = normalizeCodeText(item.itemCode);
      const nameKey = normalizeMatchText(item.itemName);
      
      let foundExisting = false;

      for (const existing of groupedItems) {
        const existingCodeKey = normalizeCodeText(existing.itemCode);
        const existingNameKey = normalizeMatchText(existing.itemName);

        let codeMatch = false;
        if (codeKey && existingCodeKey) {
          codeMatch = (codeKey === existingCodeKey);
        } else {
          // If one or both lack a code, we consider it a potential match if names align highly
          codeMatch = true;
        }

        if (codeMatch) {
          const nameSim = calculateNameSimilarity(item.itemName, existing.itemName);
          
          if (nameSim >= 0.90 || nameKey === existingNameKey) {
            foundExisting = true;

            // Sum quantities
            if (item.quantity !== null) {
              existing.quantity = (existing.quantity || 0) + item.quantity;
            }

            // Sum total prices
            if (item.totalPrice !== null) {
              existing.totalPrice = (existing.totalPrice || 0) + item.totalPrice;
            }

            // Ensure unit and unitPrice stay populated if missing in first occurrence
            if (!existing.unit && item.unit) existing.unit = item.unit;
            if (existing.unitPrice === null && item.unitPrice !== null) existing.unitPrice = item.unitPrice;
            if (!existing.itemCode && item.itemCode) existing.itemCode = item.itemCode;
            
            break;
          }
        }
      }

      if (!foundExisting) {
        groupedItems.push({ ...item });
      }
    }

    return {
      ...doc,
      lineItems: groupedItems
    };
  });

  // 1. Find base file
  let baseFile = documents[0];

  if (baseFileName) {
    const found = documents.find(d => d.fileName === baseFileName);
    if (found) {
      baseFile = found;
    } else {
      for (const doc of documents) {
        if (doc.lineItems.length > baseFile.lineItems.length) {
          baseFile = doc;
        }
      }
    }
  } else {
    for (const doc of documents) {
      if (doc.lineItems.length > baseFile.lineItems.length) {
        baseFile = doc;
      }
    }
  }

  const otherFiles = documents.filter(d => d.fileName !== baseFile.fileName);
  const results: ComparisonResult[] = [];
  const extraItems: Record<string, LineItem[]> = {};

  // 2. Get embeddings for base file items
  const baseItemNames = baseFile.lineItems.map(item => item.itemName);
  const baseEmbeddings = aiProvider === 'openai'
    ? await getEmbeddingsOpenAI(baseItemNames)
    : await getEmbeddings(baseItemNames);

  // 3. Get embeddings for other files
  const otherFilesWithEmbeddings = await Promise.all(otherFiles.map(async (file) => {
    const itemNames = file.lineItems.map(item => item.itemName);
    const embeddings = aiProvider === 'openai'
      ? await getEmbeddingsOpenAI(itemNames)
      : await getEmbeddings(itemNames);
    return { file, embeddings };
  }));

  // 4. Compare each item in base file against other files using semantic similarity and itemCode

  const allAssignments: Record<number, Record<string, { item: LineItem; score: number }[]>> = {};
  const allSuggestions: Record<number, Record<string, { item: LineItem; score: number }[]>> = {};

  for (let i = 0; i < baseFile.lineItems.length; i++) {
    allAssignments[i] = {};
    allSuggestions[i] = {};
    for (const other of otherFiles) {
      allAssignments[i][other.fileName] = [];
      allSuggestions[i][other.fileName] = [];
    }
  }

  for (let fileIdx = 0; fileIdx < otherFilesWithEmbeddings.length; fileIdx++) {
    const other = otherFilesWithEmbeddings[fileIdx];
    const scoreMatrix: { baseIdx: number; otherIdx: number; score: number }[] = [];
    const globallyAssignedOtherItems = new Set<LineItem>();

    for (let j = 0; j < other.file.lineItems.length; j++) {
      const otherItem = other.file.lineItems[j];
      const otherEmb = other.embeddings[j];

      for (let i = 0; i < baseFile.lineItems.length; i++) {
        const baseItem = baseFile.lineItems[i];
        const baseEmb = baseEmbeddings[i];

        const semanticScore = cosineSimilarity(baseEmb, otherEmb);
        const fuzzyScore = calculateNameSimilarity(baseItem.itemName, otherItem.itemName);
        const nameScore = Math.max(semanticScore, fuzzyScore);

        const basePrimaryCode = getPrimaryProductCode(baseItem);
        const otherPrimaryCode = getPrimaryProductCode(otherItem);
        const primaryCodeScore = calculateProductCodeSimilarity(basePrimaryCode, otherPrimaryCode);
        const rawCodeScore = calculateCodeSimilarity(baseItem.itemCode, otherItem.itemCode);

        let codeScore = 0;
        if (basePrimaryCode && otherPrimaryCode) {
          codeScore = primaryCodeScore;
        } else if (basePrimaryCode || otherPrimaryCode) {
          codeScore = Math.max(primaryCodeScore, rawCodeScore * 0.25);
        } else {
          codeScore = rawCodeScore * 0.2;
        }

        let finalScore = (nameScore * 0.86) + (codeScore * 0.14);

        if (primaryCodeScore >= 0.95 && nameScore >= 0.78) {
          finalScore = Math.max(finalScore, 0.92);
        } else if (primaryCodeScore >= 0.9 && nameScore >= 0.85) {
          finalScore = Math.max(finalScore, 0.9);
        }


        // Add small tie-breakers for exact numbers so it distributes better
        if (baseItem.quantity !== null && otherItem.quantity !== null && baseItem.quantity === otherItem.quantity) {
          finalScore += 0.02;
        }
        if (baseItem.unitPrice !== null && otherItem.unitPrice !== null && baseItem.unitPrice === otherItem.unitPrice) {
          finalScore += 0.01;
        }

        scoreMatrix.push({ baseIdx: i, otherIdx: j, score: finalScore });
      }
    }

    // Sort globally by score descending to get absolute best matches globally first
    scoreMatrix.sort((a, b) => b.score - a.score);

    const globallyAssignedBaseIdxs = new Set<number>();
    
    // Strict 1-to-1 Assignment (Stable Marriage approach)
    for (const match of scoreMatrix) {
      if (match.score < 0.40) continue; // Skip weak matches

      const otherItem = other.file.lineItems[match.otherIdx];
      
      // If neither the base item nor the other item has been assigned yet, bind them!
      if (!globallyAssignedBaseIdxs.has(match.baseIdx) && !globallyAssignedOtherItems.has(otherItem)) {
        allAssignments[match.baseIdx][other.file.fileName].push({
          item: otherItem,
          score: Math.min(1, match.score)
        });
        globallyAssignedBaseIdxs.add(match.baseIdx);
        globallyAssignedOtherItems.add(otherItem);
      }
    }

    // Capture extra/unassigned items
    const unassignedItemsForThisFile = other.file.lineItems.filter(item => !globallyAssignedOtherItems.has(item));
    if (unassignedItemsForThisFile.length > 0) {
      extraItems[other.file.fileName] = unassignedItemsForThisFile;
    }

    // Populate suggestions for each base item
    for (let i = 0; i < baseFile.lineItems.length; i++) {
      const scoresForBase = scoreMatrix.filter(m => m.baseIdx === i).sort((a, b) => b.score - a.score);
      const suggestions: { item: LineItem; score: number }[] = [];

      for (const s of scoresForBase) {
        const otherItem = other.file.lineItems[s.otherIdx];
        if (!globallyAssignedOtherItems.has(otherItem)) {
          suggestions.push({ item: otherItem, score: Math.min(1, s.score) });
        }
        if (suggestions.length >= 3) break;
      }
      allSuggestions[i][other.file.fileName] = suggestions;
    }
  }

  for (let i = 0; i < baseFile.lineItems.length; i++) {
    const baseItem = baseFile.lineItems[i];

    let maxRowsForThisBaseItem = 1;
    for (const other of otherFiles) {
      const assigned = allAssignments[i][other.fileName];
      // Sort assigned items by their original extraction index (physical order) to make it easier to follow
      assigned.sort((a, b) => (a.item.originalIndex || 0) - (b.item.originalIndex || 0));
      if (assigned.length > maxRowsForThisBaseItem) {
        maxRowsForThisBaseItem = assigned.length;
      }
    }

    for (let rowIdx = 0; rowIdx < maxRowsForThisBaseItem; rowIdx++) {
      const comparisons: Record<string, ComparisonDetail> = {};

      for (const other of otherFiles) {
        const assigned = allAssignments[i][other.fileName];
        const matchData = assigned[rowIdx];

        let bestMatch = matchData?.item;
        let highestScore = matchData?.score || 0;

        // Show suggestions only on the last duplicated row for this base item
        const suggestions = (rowIdx === maxRowsForThisBaseItem - 1) ? allSuggestions[i][other.fileName] : [];

        let status: MatchStatus = 'MISSING';
        const discrepancies: string[] = [];

        if (bestMatch && highestScore >= 0.40) {
          const basePrimaryCode = getPrimaryProductCode(baseItem);
          const otherPrimaryCode = getPrimaryProductCode(bestMatch);
          const itemCodeSimilarity = calculateCodeSimilarity(baseItem.itemCode, bestMatch.itemCode);
          const itemNameSimilarity = calculateNameSimilarity(baseItem.itemName, bestMatch.itemName);
          const codeCompatible = areProductCodesCompatible(baseItem, bestMatch);

          // ── Collect attribute discrepancies ──
          let hasAttributeDiscrepancy = false;

          if (activeCompareFields.includes('itemCode') && !codeCompatible) {
            if (basePrimaryCode && otherPrimaryCode) {
              hasAttributeDiscrepancy = true;
              discrepancies.push(`Mã mặt hàng khác: Gốc (${basePrimaryCode}) vs Đối chiếu (${otherPrimaryCode})`);
            } else if (itemCodeSimilarity < 0.6 && itemNameSimilarity < 0.85) {
              hasAttributeDiscrepancy = true;
              discrepancies.push(`Mã hàng lệch: Gốc (${baseItem.itemCode}) vs Đối chiếu (${bestMatch.itemCode})`);
            }
          }

          if (activeCompareFields.includes('unit') && !shouldIgnoreUnitDifference(baseItem, bestMatch)) {
            hasAttributeDiscrepancy = true;
            discrepancies.push(`Đơn vị tính lệch: Gốc (${baseItem.unit ?? 'Trống'}) vs Đối chiếu (${bestMatch.unit ?? 'Trống'})`);
          }

          if (activeCompareFields.includes('quantity') && isFieldDifferent('quantity', baseItem, bestMatch)) {
            hasAttributeDiscrepancy = true;
            discrepancies.push(`Số lượng lệch: Gốc (${baseItem.quantity ?? 'Trống'}) vs Đối chiếu (${bestMatch.quantity ?? 'Trống'})`);
          }

          if (activeCompareFields.includes('unitPrice') && isFieldDifferent('unitPrice', baseItem, bestMatch)) {
            hasAttributeDiscrepancy = true;
            discrepancies.push(`Đơn giá lệch: Gốc (${baseItem.unitPrice ?? 'Trống'}) vs Đối chiếu (${bestMatch.unitPrice ?? 'Trống'})`);
          }

          if (activeCompareFields.includes('totalPrice') && isFieldDifferent('totalPrice', baseItem, bestMatch)) {
            hasAttributeDiscrepancy = true;
            discrepancies.push(`Thành tiền lệch: Gốc (${baseItem.totalPrice ?? 'Trống'}) vs Đối chiếu (${bestMatch.totalPrice ?? 'Trống'})`);
          }

          // Name mismatch note (informational, does not force MISMATCH — identity quality is captured by score level)
          if (activeCompareFields.includes('itemName') && itemNameSimilarity < 0.75 && !codeCompatible) {
            discrepancies.push(`Tên hàng lệch: Gốc (${baseItem.itemName}) vs Đối chiếu (${bestMatch.itemName})`);
          }

          // ── Determine final status ──
          if (hasAttributeDiscrepancy) {
            status = 'MISMATCH';
          } else if (highestScore >= 0.95) {
            status = 'MATCH_PERFECT';
          } else if (highestScore >= 0.75) {
            status = 'MATCH_GOOD';
          } else if (highestScore >= 0.50) {
            status = 'MATCH_MODERATE';
          } else {
            status = 'MATCH_WEAK';
          }
        }

        comparisons[other.fileName] = {
          status,
          matchScore: Math.min(1, highestScore),
          matchedItem: bestMatch,
          discrepancies,
          suggestions: suggestions.length > 0 ? suggestions : undefined
        };
      } // Closes otherFiles loop

      results.push({
        baseItem,
        comparisons
      });
    }
  }

  // Ensure the final results list is strictly sorted by the extraction order (originalIndex)
  results.sort((a, b) => (a.baseItem.originalIndex || 0) - (b.baseItem.originalIndex || 0));

  return {
    baseFile,
    otherFiles,
    results,
    extraItems,
    compareFields: activeCompareFields,
  };
}