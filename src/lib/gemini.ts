import { GoogleGenAI, Type } from "@google/genai";
import { jsonrepair } from 'jsonrepair';
import { DocumentData } from "../types";

let aiInstance: GoogleGenAI | null = null;

function getAI(): GoogleGenAI {
  if (!aiInstance) {
    const apiKey = process.env.GEMINI_API_KEY;
    if (!apiKey) {
      throw new Error("Không tìm thấy GEMINI_API_KEY. Vui lòng cấu hình API Key trong môi trường deploy (ví dụ: GitHub Secrets).");
    }
    aiInstance = new GoogleGenAI({ apiKey });
  }
  return aiInstance;
}

const schema = {
  type: Type.OBJECT,
  properties: {
    documentType: { type: Type.STRING, description: "Loại chứng từ (VD: Đơn đặt hàng, Phiếu xuất kho, Hóa đơn)" },
    documentNumber: { type: Type.STRING, description: "Số chứng từ" },
    date: { type: Type.STRING, description: "Ngày tháng trên chứng từ" },
    lineItems: {
      type: Type.ARRAY,
      items: {
        type: Type.OBJECT,
        properties: {
          itemCode: { type: Type.STRING, description: "Mã hàng hóa, sản phẩm (Lấy từ cột Mã hàng riêng biệt nếu có, hoặc trích xuất nếu nó nằm lẫn bên trong tên sản phẩm)" },
          itemName: { type: Type.STRING, description: "Tên hàng hóa, dịch vụ" },
          quantity: { type: Type.NUMBER, description: "Số lượng" },
          unitPrice: { type: Type.NUMBER, description: "Đơn giá" },
          totalPrice: { type: Type.NUMBER, description: "Thành tiền" },
          unit: { type: Type.STRING, description: "Đơn vị tính" }
        },
        required: ["itemName"]
      }
    }
  },
  required: ["documentType", "lineItems"]
};

const GEMINI_PROMPT_SINGLE =
  "Trích xuất thông tin từ TẤT CẢ các chứng từ có trong file này (file có thể chứa nhiều trang, mỗi trang hoặc cụm trang là 1 chứng từ riêng biệt). Bao gồm loại chứng từ, số chứng từ, ngày tháng và danh sách chi tiết các mặt hàng (tên, số lượng, đơn giá, thành tiền, đơn vị tính) cho MỖI chứng từ tìm thấy.\n\nLƯU Ý QUAN TRỌNG ĐỂ KHÔNG BỎ SÓT DỮ LIỆU:\n1. Trích xuất TOÀN BỘ các dòng hàng hóa/sản phẩm có trong bảng chi tiết. KHÔNG ĐƯỢC BỎ SÓT BẤT KỲ SẢN PHẨM NÀO, hãy quét kỹ từng dòng từ trang đầu đến trang cuối.\n2. Về Mã hàng (itemCode): Ưu tiên lấy từ cột 'Mã hàng' riêng biệt. Nếu không có, hãy trích xuất mã hàng nếu nó nằm lẫn bên trong chuỗi Tên hàng hóa.\n3. CHỈ trích xuất các sản phẩm/hàng hóa thực sự. TUYỆT ĐỐI KHÔNG đưa các dòng như Tổng cộng, Chiết khấu, Thuế VAT, Phí vận chuyển vào danh sách mặt hàng.\n\nTrả về định dạng JSON chính xác là một MẢNG các chứng từ.";

function buildGeminiPromptMultiImage(count: number): string {
  return (
    `Bạn nhận ${count} ảnh theo ĐÚNG thứ tự từ trên xuống: ảnh 1 là trang đầu tiên của lô, ảnh ${count} là trang cuối của lô.\n` +
    "Trích xuất thông tin từ TẤT CẢ các chứng từ có trong TOÀN BỘ các ảnh này (mỗi ảnh có thể là một trang chứng từ). Bao gồm loại chứng từ, số chứng từ, ngày tháng và danh sách chi tiết các mặt hàng (tên, số lượng, đơn giá, thành tiền, đơn vị tính) cho MỖI chứng từ tìm thấy.\n\n" +
    "LƯU Ý QUAN TRỌNG ĐỂ KHÔNG BỎ SÓT DỮ LIỆU:\n" +
    "1. Trích xuất TOÀN BỘ các dòng hàng hóa/sản phẩm có trong bảng chi tiết trên TẤT CẢ các ảnh. KHÔNG ĐƯỢC BỎ SÓT BẤT KỲ SẢN PHẨM NÀO trên bất kỳ ảnh nào.\n" +
    "2. Về Mã hàng (itemCode): Ưu tiên lấy từ cột 'Mã hàng' riêng biệt. Nếu không có, hãy trích xuất mã hàng nếu nó nằm lẫn bên trong chuỗi Tên hàng hóa.\n" +
    "3. CHỈ trích xuất các sản phẩm/hàng hóa thực sự. TUYỆT ĐỐI KHÔNG đưa các dòng như Tổng cộng, Chiết khấu, Thuế VAT, Phí vận chuyển vào danh sách mặt hàng.\n\n" +
    "Trả về định dạng JSON chính xác là một MẢNG các chứng từ."
  );
}

function readFileAsBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => {
      if (typeof reader.result === 'string') {
        resolve(reader.result.split(',')[1]);
      } else {
        reject(new Error("Failed to read file as base64"));
      }
    };
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

/** Ghép các dòng hàng theo thứ tự AI trả về, không gộp trùng tên / không cộng dồn. */
function flattenGeminiDocuments(parsedArray: any[], logicalFileName: string): DocumentData {
  let docType = 'Không xác định';
  let docNum = 'Không xác định';
  let docDate = 'Không xác định';

  const flatItems: any[] = [];
  parsedArray.forEach((parsed: any, docIndex: number) => {
    if (docIndex === 0) {
      docType = parsed.documentType || 'Không xác định';
      docNum = parsed.documentNumber || 'Không xác định';
      docDate = parsed.date || 'Không xác định';
    }
    const items = parsed.lineItems || [];
    for (const item of items) {
      flatItems.push(item);
    }
  });

  const finalLineItems = flatItems.map((item: any, index: number) => ({
    id: `${logicalFileName}-item-${index}`,
    originalIndex: index + 1,
    itemCode: item.itemCode ?? null,
    itemName: item.itemName || 'Không xác định',
    quantity: item.quantity ?? null,
    unitPrice: item.unitPrice ?? null,
    totalPrice: item.totalPrice ?? null,
    unit: item.unit ?? null,
  }));

  console.log(`[DEBUG OCR] Danh sách dòng (không gộp) cho "${logicalFileName}":`, finalLineItems);

  return {
    fileName: logicalFileName,
    documentType: docType,
    documentNumber: docNum,
    date: docDate,
    lineItems: finalLineItems
  };
}

export async function processDocuments(files: File[], logicalFileName: string): Promise<DocumentData> {
  if (files.length === 0) {
    throw new Error('processDocuments: cần ít nhất 1 file.');
  }

  const base64List = await Promise.all(files.map((f) => readFileAsBase64(f)));
  const promptText = files.length === 1 ? GEMINI_PROMPT_SINGLE : buildGeminiPromptMultiImage(files.length);

  const contents: unknown[] = [];
  for (let i = 0; i < files.length; i++) {
    contents.push({
      inlineData: {
        data: base64List[i],
        mimeType: files[i].type || 'image/png'
      }
    });
  }
  contents.push(promptText);

  let response;
  try {
    const ai = getAI();
    response = await ai.models.generateContent({
      model: "gemini-3.1-flash-lite-preview",
      contents,
      config: {
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.ARRAY,
          items: schema
        },
        temperature: 0.1,
        maxOutputTokens: 8192,
      }
    });
  } catch (genError: any) {
    console.error("Gemini API Error:", genError);
    if (genError.message && genError.message.includes("Unterminated string in JSON")) {
      throw new Error("Dữ liệu trả về quá lớn và bị cắt ngang. Vui lòng thử chia nhỏ file PDF.");
    }
    throw new Error(`Lỗi từ AI: ${genError.message || 'Không xác định'}`);
  }

  const text = response.text;
  if (!text) throw new Error("No text returned from Gemini");

  let parsedArray;
  try {
    parsedArray = JSON.parse(text);
  } catch (error) {
    console.warn("Failed to parse JSON directly, attempting to repair...", error);
    try {
      const repairedText = jsonrepair(text);
      parsedArray = JSON.parse(repairedText);
    } catch (repairError) {
      console.error("Failed to repair JSON:", repairError);
      throw new Error("Không thể đọc dữ liệu từ AI (có thể do file quá dài hoặc định dạng lỗi).");
    }
  }

  if (!Array.isArray(parsedArray)) {
    // Fallback in case the model returns a single object instead of an array
    parsedArray = [parsedArray];
  }

  console.log(`[DEBUG OCR] Dữ liệu thô AI trả về cho "${logicalFileName}":`, parsedArray);

  return flattenGeminiDocuments(parsedArray, logicalFileName);
}

export async function processDocument(file: File, logicalFileName?: string): Promise<DocumentData> {
  return processDocuments([file], logicalFileName ?? file.name);
}

export async function getEmbeddings(texts: string[]): Promise<number[][]> {
  if (!texts || texts.length === 0) return [];
  
  const BATCH_SIZE = 100;
  const allEmbeddings: number[][] = [];
  
  for (let i = 0; i < texts.length; i += BATCH_SIZE) {
    const batch = texts.slice(i, i + BATCH_SIZE);
    const ai = getAI();
    const result = await ai.models.embedContent({
      model: 'gemini-embedding-2-preview',
      contents: batch,
    });
    
    allEmbeddings.push(...result.embeddings.map((e: any) => e.values));
  }
  
  return allEmbeddings;
}

