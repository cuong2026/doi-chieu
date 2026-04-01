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

// ─── THÊM MỚI: fuzzy name matching ───────────────────────────────────────────

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

function calculateNameSimilarity(a: string, b: string): number {
  const rawA = normalizeMatchText(a);
  const rawB = normalizeMatchText(b);

  if (!rawA || !rawB) return 0;
  if (rawA === rawB) return 1;

  const cleanA = stripLeadingCode(rawA);
  const cleanB = stripLeadingCode(rawB);

  if (cleanA === cleanB) return 0.98;

  if (cleanA.includes(cleanB) || cleanB.includes(cleanA)) return 0.93;

  const dist = levenshteinDistance(cleanA, cleanB);
  const maxLen = Math.max(cleanA.length, cleanB.length);
  return maxLen > 0 ? 1 - dist / maxLen : 0;
}

function normalizeCodeText(value: string | null | undefined): string {
  return String(value || '')
    .toUpperCase()
    .replace(/[\s_-]+/g, '')
    .trim();
}

function calculateCodeSimilarity(a: string | null | undefined, b: string | null | undefined): number {
  const codeA = normalizeCodeText(a);
  const codeB = normalizeCodeText(b);

  if (!codeA || !codeB) return 0;
  if (codeA === codeB) return 1;

  // Một mã chứa mã kia: EH2 vs EH2TI, E9838 vs E9838XA
  if (codeA.includes(codeB) || codeB.includes(codeA)) return 0.93;

  // So phần số chính: 0241 vs 241, E5382 vs 5382
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
      return normalizeText(baseItem.unit) !== normalizeText(matchedItem.unit);
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
    const itemMap = new Map<string, LineItem>();

    for (const item of validItems) {
      const codeKey = normalizeCodeText(item.itemCode);
      const nameKey = normalizeMatchText(item.itemName);
      
      // Group by itemCode primarily. If code is missing, fallback to name to avoid squashing
      const key = codeKey ? `CODE-${codeKey}` : `NAME-${nameKey}`;

      if (itemMap.has(key)) {
        const existing = itemMap.get(key)!;
        
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
      } else {
        itemMap.set(key, { ...item });
        groupedItems.push(itemMap.get(key)!);
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

    for (let j = 0; j < other.file.lineItems.length; j++) {
      const otherItem = other.file.lineItems[j];
      const otherEmb = other.embeddings[j];

      let bestBaseIdx = -1;
      let highestScore = -1;

      for (let i = 0; i < baseFile.lineItems.length; i++) {
        const baseItem = baseFile.lineItems[i];
        const baseEmb = baseEmbeddings[i];

        const semanticScore = cosineSimilarity(baseEmb, otherEmb);
        const fuzzyScore = calculateNameSimilarity(baseItem.itemName, otherItem.itemName);
        const nameScore = Math.max(semanticScore, fuzzyScore);

        let codeScore = calculateCodeSimilarity(baseItem.itemCode, otherItem.itemCode);

        let finalScore = (nameScore * 0.85) + (codeScore * 0.15);

        if (codeScore >= 1) {
          finalScore = Math.max(finalScore, 0.95);
        } else if (codeScore >= 0.9) {
          finalScore = Math.max(finalScore, 0.85);
        }

        // Add small tie-breakers for exact numbers so it distributes better
        if (baseItem.quantity !== null && otherItem.quantity !== null && baseItem.quantity === otherItem.quantity) {
          finalScore += 0.02;
        }
        if (baseItem.unitPrice !== null && otherItem.unitPrice !== null && baseItem.unitPrice === otherItem.unitPrice) {
          finalScore += 0.01;
        }

        scoreMatrix.push({ baseIdx: i, otherIdx: j, score: finalScore });

        if (finalScore > highestScore) {
          highestScore = finalScore;
          bestBaseIdx = i;
        }
      }

      if (bestBaseIdx !== -1 && highestScore >= 0.75) {
        allAssignments[bestBaseIdx][other.file.fileName].push({
          item: otherItem,
          score: Math.min(1, highestScore)
        });
      }
    }

    // Populate suggestions for each base item
    for (let i = 0; i < baseFile.lineItems.length; i++) {
      const scoresForBase = scoreMatrix.filter(m => m.baseIdx === i).sort((a, b) => b.score - a.score);
      const assignedToThis = allAssignments[i][other.file.fileName].map(a => a.item);
      const suggestions: { item: LineItem; score: number }[] = [];
      
      for (const s of scoresForBase) {
        const otherItem = other.file.lineItems[s.otherIdx];
        if (!assignedToThis.includes(otherItem)) {
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
        
        const bestMatch = matchData?.item;
        const highestScore = matchData?.score || 0;
        
        // Show suggestions only on the last duplicated row for this base item
        const suggestions = (rowIdx === maxRowsForThisBaseItem - 1) ? allSuggestions[i][other.fileName] : [];

        let status: MatchStatus = 'MISSING';
        const discrepancies: string[] = [];

        if (bestMatch && highestScore >= 0.75) {
          if (highestScore >= 0.85) {
            status = 'MATCH';
          } else {
            status = 'UNCERTAIN';
            discrepancies.push(`Tên/Mã mặt hàng khớp một phần (Độ tương đồng tổng hợp: ${Math.round(highestScore * 100)}%)`);
          }

          if (activeCompareFields.includes('itemCode')) {
            const itemCodeSimilarity = calculateCodeSimilarity(baseItem.itemCode, bestMatch.itemCode);
            const itemNameSimilarity = calculateNameSimilarity(baseItem.itemName, bestMatch.itemName);

            const shouldTreatCodeAsMatch =
              itemCodeSimilarity >= 0.8 ||
              (itemNameSimilarity >= 0.9 && itemCodeSimilarity >= 0.6);

            if (!shouldTreatCodeAsMatch) {
              status = status === 'UNCERTAIN' ? 'UNCERTAIN' : 'MISMATCH';
              discrepancies.push(`Mã hàng lệch: Gốc (${baseItem.itemCode}) vs Đối chiếu (${bestMatch.itemCode})`);
            }
          }

          if (activeCompareFields.includes('itemName')) {
            const itemNameSimilarity = calculateNameSimilarity(baseItem.itemName, bestMatch.itemName);
            if (itemNameSimilarity < 0.8) {
              status = status === 'UNCERTAIN' ? 'UNCERTAIN' : 'MISMATCH';
              discrepancies.push(`Tên hàng lệch: Gốc (${baseItem.itemName}) vs Đối chiếu (${bestMatch.itemName})`);
            }
          }

          if (activeCompareFields.includes('unit') && isFieldDifferent('unit', baseItem, bestMatch)) {
            status = status === 'UNCERTAIN' ? 'UNCERTAIN' : 'MISMATCH';
            discrepancies.push(`Đơn vị tính lệch: Gốc (${baseItem.unit ?? 'Trống'}) vs Đối chiếu (${bestMatch.unit ?? 'Trống'})`);
          }

          if (activeCompareFields.includes('quantity') && isFieldDifferent('quantity', baseItem, bestMatch)) {
            status = status === 'UNCERTAIN' ? 'UNCERTAIN' : 'MISMATCH';
            discrepancies.push(`Số lượng lệch: Gốc (${baseItem.quantity ?? 'Trống'}) vs Đối chiếu (${bestMatch.quantity ?? 'Trống'})`);
          }

          if (activeCompareFields.includes('unitPrice') && isFieldDifferent('unitPrice', baseItem, bestMatch)) {
            status = status === 'UNCERTAIN' ? 'UNCERTAIN' : 'MISMATCH';
            discrepancies.push(`Đơn giá lệch: Gốc (${baseItem.unitPrice ?? 'Trống'}) vs Đối chiếu (${bestMatch.unitPrice ?? 'Trống'})`);
          }

          if (activeCompareFields.includes('totalPrice') && isFieldDifferent('totalPrice', baseItem, bestMatch)) {
            status = status === 'UNCERTAIN' ? 'UNCERTAIN' : 'MISMATCH';
            discrepancies.push(`Thành tiền lệch: Gốc (${baseItem.totalPrice ?? 'Trống'}) vs Đối chiếu (${bestMatch.totalPrice ?? 'Trống'})`);
          }
        }

        comparisons[other.fileName] = {
          status,
          matchedItem: bestMatch, // Removed redundant threshold check since it's already verified and pushed conditionally
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
    compareFields: activeCompareFields,
  };
}