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
  additionalProperties: false,
  properties: {
    documentType: { type: "string", description: "Loại chứng từ (VD: Đơn đặt hàng, Phiếu xuất kho, Hóa đơn)" },
    documentNumber: { type: ["string", "null"], description: "Số chứng từ" },
    date: { type: ["string", "null"], description: "Ngày tháng trên chứng từ" },
    lineItems: {
      type: "array",
      items: {
        type: "object",
        additionalProperties: false,
        properties: {
          itemCode: { type: ["string", "null"], description: "Mã hàng hóa, sản phẩm (Lấy từ cột Mã hàng riêng biệt nếu có, hoặc trích xuất nếu nó nằm lẫn bên trong tên sản phẩm)" },
          itemName: { type: "string", description: "Tên hàng hóa, dịch vụ" },
          quantity: { type: ["number", "null"], description: "Số lượng" },
          unitPrice: { type: ["number", "null"], description: "Đơn giá" },
          totalPrice: { type: ["number", "null"], description: "Thành tiền" },
          unit: { type: ["string", "null"], description: "Đơn vị tính" }
        },
        required: ["itemCode", "itemName", "quantity", "unitPrice", "totalPrice", "unit"]
      }
    }
  },
  required: ["documentType", "documentNumber", "date", "lineItems"]
};

const OPENAI_PROMPT_SINGLE =
  "Trích xuất thông tin từ TẤT CẢ các chứng từ có trong file này (file có thể chứa nhiều trang, mỗi trang hoặc cụm trang là 1 chứng từ riêng biệt). Bao gồm loại chứng từ, số chứng từ, ngày tháng và danh sách chi tiết các mặt hàng (tên, số lượng, đơn giá, thành tiền, đơn vị tính) cho MỖI chứng từ tìm thấy.\n\nLƯU Ý QUAN TRỌNG ĐỂ KHÔNG BỎ SÓT DỮ LIỆU:\n1. Trích xuất TOÀN BỘ các dòng hàng hóa/sản phẩm có trong bảng chi tiết. KHÔNG ĐƯỢC BỎ SÓT BẤT KỲ SẢN PHẨM NÀO, hãy quét kỹ từng dòng từ trang đầu đến trang cuối.\n2. Về Mã hàng (itemCode): Ưu tiên lấy từ cột 'Mã hàng' riêng biệt. Nếu không có, hãy trích xuất mã hàng nếu nó nằm lẫn bên trong chuỗi Tên hàng hóa.\n3. CHỈ trích xuất các sản phẩm/hàng hóa thực sự. TUYỆT ĐỐI KHÔNG đưa các dòng như Tổng cộng, Chiết khấu, Thuế VAT, Phí vận chuyển vào danh sách mặt hàng.\n\nTrả về định dạng JSON chính xác là một MẢNG các chứng từ.";

function buildOpenAIPromptMultiImage(count: number): string {
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

function readFileAsDataURL(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => {
      if (typeof reader.result === 'string') {
        resolve(reader.result);
      } else {
        reject(new Error("Failed to read file as base64"));
      }
    };
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

/** Ghép các dòng hàng theo thứ tự AI trả về, không gộp trùng tên / không cộng dồn. */
function flattenOpenAIDocuments(parsedArray: any[], logicalFileName: string): DocumentData {
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

export async function processDocumentsOpenAI(files: File[], logicalFileName: string): Promise<DocumentData> {
  if (files.length === 0) {
    throw new Error('processDocumentsOpenAI: cần ít nhất 1 file.');
  }

  const dataUrls = await Promise.all(files.map((f) => readFileAsDataURL(f)));
  const promptText = files.length === 1 ? OPENAI_PROMPT_SINGLE : buildOpenAIPromptMultiImage(files.length);

  const content: Array<
    | { type: 'text'; text: string }
    | { type: 'image_url'; image_url: { url: string } }
  > = [{ type: 'text', text: promptText }];
  for (const url of dataUrls) {
    content.push({ type: 'image_url', image_url: { url } });
  }

  let response;
  try {
    const openai = getOpenAI();
    response = await openai.chat.completions.create({
      model: "gpt-5.4-mini-2026-03-17",
      messages: [
        {
          role: "user",
          content
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
      temperature: 1,
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

  console.log(`[DEBUG OCR] Dữ liệu thô AI trả về cho "${logicalFileName}":`, parsedArray);

  return flattenOpenAIDocuments(parsedArray, logicalFileName);
}

export async function processDocumentOpenAI(file: File, logicalFileName?: string): Promise<DocumentData> {
  return processDocumentsOpenAI([file], logicalFileName ?? file.name);
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
