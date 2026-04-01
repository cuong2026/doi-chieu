import { DocumentData, LineItem, ReportData, ComparisonResult, ComparisonDetail, MatchStatus, AIProvider } from '../types';
import { getEmbeddings } from './gemini';
import { getEmbeddingsOpenAI } from './openai';

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

export async function generateReport(rawDocuments: DocumentData[], baseFileName?: string | null, aiProvider: AIProvider = 'openai'): Promise<ReportData> {
  if (rawDocuments.length === 0) throw new Error("No documents provided");

  // Filter out non-products to ensure accurate base file selection and clean reports
  // A valid product should have at least one numeric field (quantity, unitPrice, or totalPrice)
  // We also exclude common non-product keywords just in case AI includes them.
  const documents = rawDocuments.map(doc => ({
    ...doc,
    lineItems: doc.lineItems.filter(item => {
      const hasSomeNumber = item.quantity !== null || item.unitPrice !== null || item.totalPrice !== null;
      const isExcluded = /^(tổng|cộng|chiết khấu|thuế|vat|tiền hàng|giảm giá|phí|thanh toán)/i.test(item.itemName.trim());
      return hasSomeNumber && !isExcluded;
    })
  }));

  // 1. Find base file
  let baseFile = documents[0];
  
  if (baseFileName) {
    const found = documents.find(d => d.fileName === baseFileName);
    if (found) {
      baseFile = found;
    } else {
      // Fallback to max line items if not found
      for (const doc of documents) {
        if (doc.lineItems.length > baseFile.lineItems.length) {
          baseFile = doc;
        }
      }
    }
  } else {
    // Default: max valid line items
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
  for (let i = 0; i < baseFile.lineItems.length; i++) {
    const baseItem = baseFile.lineItems[i];
    const baseEmb = baseEmbeddings[i];
    const comparisons: Record<string, ComparisonDetail> = {};

    for (const other of otherFilesWithEmbeddings) {
      const scoredItems: { item: LineItem; score: number }[] = [];

      for (let j = 0; j < other.file.lineItems.length; j++) {
        const item = other.file.lineItems[j];
        const itemEmb = other.embeddings[j];
        
        // 1. So sánh tên qua ngữ nghĩa -> %
        const nameScore = cosineSimilarity(baseEmb, itemEmb);
        
        // 2. So sánh mã hàng (Yếu tố bổ trợ) -> %
        const baseCode = baseItem.itemCode?.trim().toLowerCase() || '';
        const matchCode = item.itemCode?.trim().toLowerCase() || '';
        const baseName = baseItem.itemName.toLowerCase();
        const matchName = item.itemName.toLowerCase();

        let codeScore = nameScore; // Mặc định trung lập (bằng điểm tên) để không ảnh hưởng nếu thiếu mã
        
        if (baseCode && matchCode) {
          if (baseCode === matchCode) {
            codeScore = 1.0; // Khớp hoàn toàn
          } else if (baseCode.includes(matchCode) || matchCode.includes(baseCode)) {
            codeScore = 0.8; // Khớp một phần
          } else {
            codeScore = 0.4; // Khác mã (có thể do lỗi đánh máy), không cho 0 để tránh kéo tụt điểm tên chuẩn
          }
        } else if (baseCode && !matchCode) {
          if (matchName.includes(baseCode)) codeScore = 0.9;
        } else if (!baseCode && matchCode) {
          if (baseName.includes(matchCode)) codeScore = 0.9;
        }

        // 3. Tính điểm tổng hợp: Tên chiếm 85% (Quyết định chính), Mã chiếm 15% (Bổ trợ)
        let finalScore = (nameScore * 0.85) + (codeScore * 0.15);

        // 4. ĐẶC CÁCH (Override): Nếu mã hàng khớp chính xác 100%, hoặc mã hàng bên này xuất hiện rõ ràng trong tên bên kia
        // Đây là bằng chứng rất mạnh cho thấy chúng là cùng 1 sản phẩm dù tên gọi (ngữ nghĩa) khác xa nhau.
        if (baseCode && matchCode && baseCode === matchCode) {
          finalScore = Math.max(finalScore, 0.95); // Đảm bảo chắc chắn MATCH
        } else if (baseCode && matchName.includes(baseCode)) {
          finalScore = Math.max(finalScore, 0.85); // Đẩy lên ngưỡng MATCH
        } else if (matchCode && baseName.includes(matchCode)) {
          finalScore = Math.max(finalScore, 0.85); // Đẩy lên ngưỡng MATCH
        }
        
        scoredItems.push({ item, score: finalScore });
      }

      // Sort items by score descending
      scoredItems.sort((a, b) => b.score - a.score);

      const bestMatchData = scoredItems.length > 0 ? scoredItems[0] : null;
      const bestMatch = bestMatchData?.item;
      const highestScore = bestMatchData?.score || 0;
      
      // Get top 3 suggestions (excluding the best match if it's considered a match)
      const suggestions = scoredItems.slice(1, 4);

      let status: MatchStatus = 'MISSING';
      const discrepancies: string[] = [];

      // Ngưỡng đánh giá dựa trên điểm trung bình
      if (bestMatch && highestScore > 0.75) { 
        if (highestScore >= 0.85) {
          status = 'MATCH';
        } else {
          status = 'UNCERTAIN';
          discrepancies.push(`Tên/Mã mặt hàng khớp một phần (Độ tương đồng tổng hợp: ${Math.round(highestScore * 100)}%)`);
        }

        // Check code discrepancies
        const baseCode = baseItem.itemCode?.trim().toLowerCase();
        const matchCode = bestMatch.itemCode?.trim().toLowerCase();
        
        if (baseCode && matchCode && baseCode !== matchCode) {
          status = status === 'UNCERTAIN' ? 'UNCERTAIN' : 'MISMATCH';
          discrepancies.push(`Mã hàng lệch: Gốc (${baseItem.itemCode}) vs Đối chiếu (${bestMatch.itemCode})`);
        } else if (baseCode && !matchCode && !bestMatch.itemName.toLowerCase().includes(baseCode)) {
          discrepancies.push(`Không tìm thấy mã hàng (${baseItem.itemCode}) trong đối chiếu`);
        } else if (!baseCode && matchCode && !baseItem.itemName.toLowerCase().includes(matchCode)) {
          discrepancies.push(`Đối chiếu có mã (${bestMatch.itemCode}) nhưng gốc không có`);
        }

        // Compare fields
if ((baseItem.unit || '') !== (bestMatch.unit || '')) {
  status = status === 'UNCERTAIN' ? 'UNCERTAIN' : 'MISMATCH';
  discrepancies.push(`Đơn vị tính lệch: Gốc (${baseItem.unit ?? 'Trống'}) vs Đối chiếu (${bestMatch.unit ?? 'Trống'})`);
}

if (baseItem.quantity !== bestMatch.quantity) {
  status = status === 'UNCERTAIN' ? 'UNCERTAIN' : 'MISMATCH';
  discrepancies.push(`Số lượng lệch: Gốc (${baseItem.quantity ?? 'Trống'}) vs Đối chiếu (${bestMatch.quantity ?? 'Trống'})`);
}

if (baseItem.unitPrice !== bestMatch.unitPrice) {
  status = status === 'UNCERTAIN' ? 'UNCERTAIN' : 'MISMATCH';
  discrepancies.push(`Đơn giá lệch: Gốc (${baseItem.unitPrice ?? 'Trống'}) vs Đối chiếu (${bestMatch.unitPrice ?? 'Trống'})`);
}

if (baseItem.totalPrice !== bestMatch.totalPrice) {
  status = status === 'UNCERTAIN' ? 'UNCERTAIN' : 'MISMATCH';
  discrepancies.push(`Thành tiền lệch: Gốc (${baseItem.totalPrice ?? 'Trống'}) vs Đối chiếu (${bestMatch.totalPrice ?? 'Trống'})`);
}
      }

      comparisons[other.file.fileName] = {
        status,
        matchedItem: bestMatch && highestScore > 0.75 ? bestMatch : undefined,
        discrepancies,
        suggestions: suggestions.length > 0 ? suggestions : undefined
      };
    }

    results.push({
      baseItem,
      comparisons
    });
  }

  return {
    baseFile,
    otherFiles,
    results
  };
}
