import OpenAI from "openai";
import { jsonrepair } from 'jsonrepair';
import { DocumentData, ItemCodeLocation } from "../types";

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
            itemCode: { type: ["string", "null"], description: itemCodeDesc },
            itemName: { type: "string", description: "Tên hàng hóa, dịch vụ (Thường là cột chứa chuỗi dài, mô tả có nghĩa)" },
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

function buildOpenAIPromptSingle(itemCodeLocation: ItemCodeLocation): string {
  return "Trích xuất thông tin từ TẤT CẢ các chứng từ có trong file này (file có thể chứa nhiều trang, mỗi trang hoặc cụm trang là 1 chứng từ riêng biệt). Bao gồm loại chứng từ, số chứng từ, ngày tháng và danh sách chi tiết các mặt hàng (tên, số lượng, đơn giá, thành tiền, đơn vị tính) cho MỖI chứng từ tìm thấy.\n\nLƯU Ý QUAN TRỌNG ĐỂ KHÔNG BỎ SÓT DỮ LIỆU:\n1. Hỗ Trợ Nhận Diện Cột (Cho trang mất tiêu đề): 'Tên hàng hóa' là chuỗi dài, có nghĩa mô tả sản phẩm. 'Mã hàng' thường là ký tự ngắn, viết hoa hoặc số.\n2. Trích xuất TOÀN BỘ các dòng hàng hóa/sản phẩm có trong bảng chi tiết. KHÔNG ĐƯỢC BỎ SÓT BẤT KỲ SẢN PHẨM NÀO, hãy quét kỹ từng dòng từ trang đầu đến trang cuối.\n3. Về " + getItemCodePrompt(itemCodeLocation) + "\n4. CHỈ trích xuất các sản phẩm/hàng hóa thực sự. TUYỆT ĐỐI KHÔNG đưa các dòng như Tổng cộng, Chiết khấu, Thuế VAT, Phí vận chuyển vào danh sách mặt hàng.\n\nTrả về định dạng JSON chính xác là một MẢNG các chứng từ.";
}

function buildOpenAIPromptMultiImage(count: number, itemCodeLocation: ItemCodeLocation): string {
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
/**
 * Khi chế độ là 'in_name', dùng regex để tự cắt mã sản phẩm từ chuỗi tên hàng,
 * vì AI vẫn cứng đầu lấy từ cột mã riêng dù prompt nói không.
 */
function extractCodeFromName(name: string): string | null {
  if (!name) return null;
  const m1 = name.match(/^([A-Za-z]{0,5}\d[\w-]*?)[_\-\s]/);
  if (m1) return m1[1];
  const m2 = name.match(/^([A-Za-z]*\d+)\s/);
  if (m2) return m2[1];
  return null;
}

function flattenOpenAIDocuments(parsedArray: any[], logicalFileName: string, itemCodeLocation: ItemCodeLocation = 'auto'): DocumentData {
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

export async function processDocumentsOpenAI(files: File[], logicalFileName: string, itemCodeLocation: ItemCodeLocation = 'auto'): Promise<DocumentData> {
  if (files.length === 0) {
    throw new Error('processDocumentsOpenAI: cần ít nhất 1 file.');
  }

  const dataUrls = await Promise.all(files.map((f) => readFileAsDataURL(f)));
  const promptText = files.length === 1 ? buildOpenAIPromptSingle(itemCodeLocation) : buildOpenAIPromptMultiImage(files.length, itemCodeLocation);

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
                items: getSchema(itemCodeLocation)
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

  return flattenOpenAIDocuments(parsedArray, logicalFileName, itemCodeLocation);
}

export async function processDocumentOpenAI(file: File, logicalFileName?: string, itemCodeLocation: ItemCodeLocation = 'auto'): Promise<DocumentData> {
  return processDocumentsOpenAI([file], logicalFileName ?? file.name, itemCodeLocation);
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
