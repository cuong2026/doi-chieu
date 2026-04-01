import OpenAI from "openai";
import { jsonrepair } from 'jsonrepair';
import { DocumentData } from "../types";

let openaiInstance: OpenAI | null = null;

function getOpenAI(): OpenAI {
  if (!openaiInstance) {
    const apiKey = process.env.OPENAI_API_KEY;
    if (!apiKey) {
      throw new Error("Không tìm thấy OPENAI_API_KEY. Vui lòng cấu hình API Key trong môi trường deploy.");
    }
    openaiInstance = new OpenAI({ apiKey, dangerouslyAllowBrowser: true });
  }
  return openaiInstance;
}

const schema = {
  type: "object",
  properties: {
    documentType: { type: "string", description: "Loại chứng từ (VD: Đơn đặt hàng, Phiếu xuất kho, Hóa đơn)" },
    documentNumber: { type: "string", description: "Số chứng từ" },
    date: { type: "string", description: "Ngày tháng trên chứng từ" },
    lineItems: {
      type: "array",
      items: {
        type: "object",
        properties: {
          itemCode: { type: "string", description: "Mã hàng hóa, sản phẩm (Lấy từ cột Mã hàng riêng biệt nếu có, hoặc trích xuất nếu nó nằm lẫn bên trong tên sản phẩm)" },
          itemName: { type: "string", description: "Tên hàng hóa, dịch vụ" },
          quantity: { type: "number", description: "Số lượng" },
          unitPrice: { type: "number", description: "Đơn giá" },
          totalPrice: { type: "number", description: "Thành tiền" },
          unit: { type: "string", description: "Đơn vị tính" }
        },
        required: ["itemName"]
      }
    }
  },
  required: ["documentType", "lineItems"]
};

export async function processDocumentOpenAI(file: File): Promise<DocumentData> {
  const base64EncodedDataPromise = new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => {
      if (typeof reader.result === 'string') {
        resolve(reader.result); // Keep the data:image/... prefix for OpenAI
      } else {
        reject(new Error("Failed to read file as base64"));
      }
    };
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });

  const base64Data = await base64EncodedDataPromise;

  let response;
  try {
    const openai = getOpenAI();
    response = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [
        {
          role: "user",
          content: [
            {
              type: "text",
              text: "Trích xuất thông tin từ TẤT CẢ các chứng từ có trong file này (file có thể chứa nhiều trang, mỗi trang hoặc cụm trang là 1 chứng từ riêng biệt). Bao gồm loại chứng từ, số chứng từ, ngày tháng và danh sách chi tiết các mặt hàng (tên, số lượng, đơn giá, thành tiền, đơn vị tính) cho MỖI chứng từ tìm thấy.\n\nLƯU Ý QUAN TRỌNG ĐỂ KHÔNG BỎ SÓT DỮ LIỆU:\n1. Trích xuất TOÀN BỘ các dòng hàng hóa/sản phẩm có trong bảng chi tiết. KHÔNG ĐƯỢC BỎ SÓT BẤT KỲ SẢN PHẨM NÀO, hãy quét kỹ từng dòng từ trang đầu đến trang cuối.\n2. Về Mã hàng (itemCode): Ưu tiên lấy từ cột 'Mã hàng' riêng biệt. Nếu không có, hãy trích xuất mã hàng nếu nó nằm lẫn bên trong chuỗi Tên hàng hóa.\n3. CHỈ trích xuất các sản phẩm/hàng hóa thực sự. TUYỆT ĐỐI KHÔNG đưa các dòng như Tổng cộng, Chiết khấu, Thuế VAT, Phí vận chuyển vào danh sách mặt hàng.\n\nTrả về định dạng JSON chính xác là một MẢNG các chứng từ."
            },
            {
              type: "image_url",
              image_url: {
                url: base64Data
              }
            }
          ]
        }
      ],
      response_format: {
        type: "json_schema",
        json_schema: {
          name: "documents",
          schema: {
            type: "object",
            properties: {
              documents: {
                type: "array",
                items: schema
              }
            },
            required: ["documents"],
            additionalProperties: false
          },
          strict: true
        }
      },
      temperature: 0.1,
    });
  } catch (genError: any) {
    console.error("OpenAI API Error:", genError);
    throw new Error(`Lỗi từ AI: ${genError.message || 'Không xác định'}`);
  }

  const text = response.choices[0]?.message?.content;
  if (!text) throw new Error("No text returned from OpenAI");

  let parsedArray;
  try {
    const parsedObj = JSON.parse(text);
    parsedArray = parsedObj.documents;
  } catch (error) {
    console.warn("Failed to parse JSON directly, attempting to repair...", error);
    try {
      const repairedText = jsonrepair(text);
      const parsedObj = JSON.parse(repairedText);
      parsedArray = parsedObj.documents;
    } catch (repairError) {
      console.error("Failed to repair JSON:", repairError);
      throw new Error("Không thể đọc dữ liệu từ AI (có thể do file quá dài hoặc định dạng lỗi).");
    }
  }

  if (!Array.isArray(parsedArray)) {
    parsedArray = [parsedArray];
  }

  console.log(`[DEBUG OCR] Dữ liệu thô AI trả về cho file "${file.name}":`, parsedArray);

  // Aggregate items across all documents found in the file
  const aggregatedItems = new Map<string, any>();
  let docType = 'Không xác định';
  let docNum = 'Không xác định';
  let docDate = 'Không xác định';

  parsedArray.forEach((parsed: any, docIndex: number) => {
    // Take metadata from the first document
    if (docIndex === 0) {
      docType = parsed.documentType || 'Không xác định';
      docNum = parsed.documentNumber || 'Không xác định';
      docDate = parsed.date || 'Không xác định';
    }

    const items = parsed.lineItems || [];
    items.forEach((item: any) => {
      // Create a unique key based on name and unit price to aggregate quantities
      const key = `${(item.itemName || '').trim().toLowerCase()}_${item.unitPrice || 0}`;
      
      if (aggregatedItems.has(key)) {
        const existingItem = aggregatedItems.get(key);
        // Sum quantities
        existingItem.quantity = (existingItem.quantity || 0) + (item.quantity || 0);
        // Recalculate total price if possible
        if (existingItem.quantity && existingItem.unitPrice) {
          existingItem.totalPrice = existingItem.quantity * existingItem.unitPrice;
        }
      } else {
        aggregatedItems.set(key, { ...item });
      }
    });
  });

  const finalLineItems = Array.from(aggregatedItems.values()).map((item: any, index: number) => ({
    id: `${file.name}-item-${index}`,
    originalIndex: index + 1,
    itemCode: item.itemCode ?? null,
    itemName: item.itemName || 'Không xác định',
    quantity: item.quantity ?? null,
    unitPrice: item.unitPrice ?? null,
    totalPrice: item.totalPrice ?? null,
    unit: item.unit ?? null,
  }));

  console.log(`[DEBUG OCR] Dữ liệu sau khi gộp (Aggregated) cho file "${file.name}":`, finalLineItems);

  return {
    fileName: file.name,
    documentType: docType,
    documentNumber: docNum,
    date: docDate,
    lineItems: finalLineItems
  };
}

export async function getEmbeddingsOpenAI(texts: string[]): Promise<number[][]> {
  if (!texts || texts.length === 0) return [];
  
  const BATCH_SIZE = 100;
  const allEmbeddings: number[][] = [];
  
  for (let i = 0; i < texts.length; i += BATCH_SIZE) {
    const batch = texts.slice(i, i + BATCH_SIZE);
    const openai = getOpenAI();
    const result = await openai.embeddings.create({
      model: 'text-embedding-3-small',
      input: batch,
    });
    
    allEmbeddings.push(...result.data.map((e: any) => e.embedding));
  }
  
  return allEmbeddings;
}
