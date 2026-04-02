import { GoogleGenAI, Type } from "@google/genai";
import { jsonrepair } from 'jsonrepair';
import { DocumentData, ItemCodeLocation } from "../types";

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

function getSchema(location: ItemCodeLocation) {
  let itemCodeDesc = "Mã hàng hóa, sản phẩm";
  if (location === 'separate_column') {
    itemCodeDesc = "Mã hàng hóa, sản phẩm (CHỈ copy y nguyên từ cột Mã hàng riêng biệt, KHÔNG trích xuất mã ngầm từ trong cột tên)";
  } else if (location === 'in_name') {
    itemCodeDesc = "Mã hàng hóa, sản phẩm (***QUAN TRỌNG: Lấy mã từ BÊN TRONG chuỗi Tên hàng hóa. TUYỆT ĐỐI BỎ QUA các cột mã đứng riêng lẻ ngoài bảng.*** Dữ liệu itemCode PHẢI là chuỗi con rút ra từ itemName)";
  } else {
    itemCodeDesc = "Mã hàng hóa, sản phẩm (Ưu tiên lấy từ cột Mã hàng riêng biệt, nếu không có thì trích xuất từ bên trong tên sản phẩm)";
  }

  return {
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
            itemCode: { type: Type.STRING, description: itemCodeDesc },
            itemName: { type: Type.STRING, description: "Tên hàng hóa, dịch vụ (Thường là cột chứa chuỗi dài, mô tả có nghĩa)" },
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
}

function getItemCodePrompt(location: ItemCodeLocation): string {
  if (location === 'separate_column') {
    return "Mã hàng (itemCode): CHỈ lấy từ cột 'Mã hàng' riêng biệt (thường là cột rỗng hoặc ký hiệu ngắn). KHÔNG trích xuất chui từ trong Tên hàng.";
  } else if (location === 'in_name') {
    return "Mã hàng (itemCode): ***BẮT BUỘC BỎ QUA CÁC CỘT CHỨA MÃ ĐỨNG RIÊNG LẺ***. Thay vào đó, bạn PHẢI đọc cột 'Tên hàng' (cột có chuỗi text mô tả rất dài) và tự cắt/trích xuất mã sản phẩm rớt ra từ chuỗi Tên hàng đó. VD: Tên hàng là '5382_File hồ sơ', bạn trả về itemCode = '5382' và itemName = '5382_File hồ sơ'.  Hãy CẨN THẬN khi không thấy tiêu đề cột ở các trang sau, đừng lầm tưởng cái cột mã ngắn ngủn đứng riêng kia là itemCode cần lấy!";
  } else {
    return "Mã hàng (itemCode): Ưu tiên lấy từ cột 'Mã hàng' riêng biệt. Nếu không có cột riêng, hãy trích xuất mã hàng nếu nó nằm lẫn bên trong chuỗi Tên hàng hóa.";
  }
}

function buildGeminiPromptSingle(itemCodeLocation: ItemCodeLocation): string {
  return "Trích xuất thông tin từ TẤT CẢ các chứng từ có trong file này (file có thể chứa nhiều trang, mỗi trang hoặc cụm trang là 1 chứng từ riêng biệt). Bao gồm loại chứng từ, số chứng từ, ngày tháng và danh sách chi tiết các mặt hàng (tên, số lượng, đơn giá, thành tiền, đơn vị tính) cho MỖI chứng từ tìm thấy.\n\nLƯU Ý QUAN TRỌNG ĐỂ KHÔNG BỎ SÓT DỮ LIỆU:\n1. Hỗ Trợ Nhận Diện Cột (Cho trang mất tiêu đề): 'Tên hàng hóa' là chuỗi dài, có nghĩa mô tả sản phẩm. 'Mã hàng' thường là ký tự ngắn, viết hoa hoặc số.\n2. Trích xuất TOÀN BỘ các dòng hàng hóa/sản phẩm có trong bảng chi tiết. KHÔNG ĐƯỢC BỎ SÓT BẤT KỲ SẢN PHẨM NÀO, hãy quét kỹ từng dòng từ trang đầu đến trang cuối.\n3. Về " + getItemCodePrompt(itemCodeLocation) + "\n4. CHỈ trích xuất các sản phẩm/hàng hóa thực sự. TUYỆT ĐỐI KHÔNG đưa các dòng như Tổng cộng, Chiết khấu, Thuế VAT, Phí vận chuyển vào danh sách mặt hàng.\n\nTrả về định dạng JSON chính xác là một MẢNG các chứng từ.";
}

function buildGeminiPromptMultiImage(count: number, itemCodeLocation: ItemCodeLocation): string {
  return (
    `Bạn nhận ${count} ảnh theo ĐÚNG thứ tự từ trên xuống: ảnh 1 là trang đầu tiên của lô, ảnh ${count} là trang cuối của lô.\n` +
    "Trích xuất thông tin từ TẤT CẢ các chứng từ có trong TOÀN BỘ các ảnh này (mỗi ảnh có thể là một trang chứng từ). Bao gồm loại chứng từ, số chứng từ, ngày tháng và danh sách chi tiết các mặt hàng (tên, số lượng, đơn giá, thành tiền, đơn vị tính) cho MỖI chứng từ tìm thấy.\n\n" +
    "LƯU Ý QUAN TRỌNG ĐỂ KHÔNG BỎ SÓT DỮ LIỆU:\n" +
    "1. Hỗ Trợ Nhận Diện Cột (Cho trang mất tiêu đề): 'Tên hàng hóa' là chuỗi dài, có nghĩa mô tả sản phẩm. 'Mã hàng' thường là ký tự ngắn, số, viết hoa (ví dụ: E5382-XD, ...).\n" +
    "2. Trích xuất TOÀN BỘ các dòng hàng hóa/sản phẩm có trong bảng chi tiết trên TẤT CẢ các ảnh. KHÔNG ĐƯỢC BỎ SÓT BẤT KỲ SẢN PHẨM NÀO trên bất kỳ ảnh nào.\n" +
    "3. Về " + getItemCodePrompt(itemCodeLocation) + "\n" +
    "4. CHỈ trích xuất các sản phẩm/hàng hóa thực sự. TUYỆT ĐỐI KHÔNG đưa các dòng như Tổng cộng, Chiết khấu, Thuế VAT, Phí vận chuyển vào danh sách mặt hàng.\n\n" +
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

/**
 * Khi chế độ là 'in_name', dùng regex để tự cắt mã sản phẩm từ chuỗi tên hàng,
 * vì AI vẫn cứng đầu lấy từ cột mã riêng dù prompt nói không.
 * Nhận dạng các mẫu:
 *   "E5382_File hồ sơ_A4"  → itemCode = "E5382"
 *   "5382_File hồ sơ"      → itemCode = "5382"
 *   "EA047_Giấy nhán 76x126" → itemCode = "EA047"
 */
function extractCodeFromName(name: string): string | null {
  if (!name) return null;
  // Mẫu 1: Mã đứng đầu, theo sau bởi dấu gạch dưới hoặc dấu gạch ngang
  const m1 = name.match(/^([A-Za-z]{0,5}\d[\w-]*?)[_\-\s]/);
  if (m1) return m1[1];
  // Mẫu 2: Mã thuần số/chữ + số đứng đầu
  const m2 = name.match(/^([A-Za-z]*\d+)\s/);
  if (m2) return m2[1];
  return null;
}

/** Ghép các dòng hàng theo thứ tự AI trả về, không gộp trùng tên / không cộng dồn. */
function flattenGeminiDocuments(parsedArray: any[], logicalFileName: string, itemCodeLocation: ItemCodeLocation = 'auto'): DocumentData {
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

  // Hậu xử lý: Nếu chế độ là 'in_name', ép lấy mã từ itemName bằng regex
  if (itemCodeLocation === 'in_name') {
    for (const item of finalLineItems) {
      const codeFromName = extractCodeFromName(item.itemName);
      if (codeFromName) {
        item.itemCode = codeFromName;
      }
    }
    console.log(`[DEBUG POST-PROCESS] Đã ép trích xuất mã từ tên cho "${logicalFileName}" (in_name mode):`, finalLineItems.map(i => `${i.itemName} → ${i.itemCode}`));
  }

  console.log(`[DEBUG OCR] Danh sách dòng (không gộp) cho "${logicalFileName}":`, finalLineItems);

  return {
    fileName: logicalFileName,
    documentType: docType,
    documentNumber: docNum,
    date: docDate,
    lineItems: finalLineItems
  };
}

export async function processDocuments(files: File[], logicalFileName: string, itemCodeLocation: ItemCodeLocation = 'auto'): Promise<DocumentData> {
  if (files.length === 0) {
    throw new Error('processDocuments: cần ít nhất 1 file.');
  }

  const base64List = await Promise.all(files.map((f) => readFileAsBase64(f)));
  const promptText = files.length === 1 ? buildGeminiPromptSingle(itemCodeLocation) : buildGeminiPromptMultiImage(files.length, itemCodeLocation);

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
          items: getSchema(itemCodeLocation)
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

  return flattenGeminiDocuments(parsedArray, logicalFileName, itemCodeLocation);
}

export async function processDocument(file: File, logicalFileName?: string, itemCodeLocation: ItemCodeLocation = 'auto'): Promise<DocumentData> {
  return processDocuments([file], logicalFileName ?? file.name, itemCodeLocation);
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

