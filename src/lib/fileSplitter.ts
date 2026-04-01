import { PDFDocument } from 'pdf-lib';
import { getDocument, GlobalWorkerOptions } from 'pdfjs-dist';
import pdfWorkerUrl from 'pdfjs-dist/build/pdf.worker.min.mjs?url';
import * as XLSX from 'xlsx';

GlobalWorkerOptions.workerSrc = pdfWorkerUrl;

export type SplitMode = 'pdf-page-images' | 'pdf-chunk' | 'excel-chunk' | 'passthrough';

export interface SplitChunkMeta {
  chunkIndex: number;
  totalChunks: number;
  pageStart?: number;
  pageEnd?: number;
}

export interface SplitResult {
  chunks: File[];
  mode: SplitMode;
  sourcePageCount?: number;
  processedPageCount?: number;
  chunkMeta: SplitChunkMeta[];
}

export async function splitFile(file: File, maxPages: number = 10, maxRows: number = 500): Promise<File[]> {
  const result = await splitFileWithMetadata(file, maxPages, maxRows);
  return result.chunks;
}

export async function splitFileWithMetadata(file: File, maxPages: number = 10, maxRows: number = 500): Promise<SplitResult> {
  const type = file.type;
  const extension = file.name.split('.').pop()?.toLowerCase();

  if (type === 'application/pdf' || extension === 'pdf') {
    try {
      const pageImageResult = await splitPdfToPageImages(file);
      if (
        pageImageResult.sourcePageCount &&
        pageImageResult.processedPageCount &&
        pageImageResult.sourcePageCount !== pageImageResult.processedPageCount
      ) {
        throw new Error(
          `Số trang render (${pageImageResult.processedPageCount}) khác số trang gốc (${pageImageResult.sourcePageCount}).`
        );
      }
      return pageImageResult;
    } catch (error) {
      console.warn('[PDF SPLIT] Render PDF thành ảnh thất bại, fallback sang cắt PDF theo chunk.', error);
      return splitPdf(file, maxPages);
    }
  } else if (
    type === 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' ||
    type === 'application/vnd.ms-excel' ||
    extension === 'xlsx' ||
    extension === 'xls' ||
    extension === 'csv'
  ) {
    return splitExcel(file, maxRows);
  }

  // For other types (images, etc.), no splitting for now
  return {
    chunks: [file],
    mode: 'passthrough',
    chunkMeta: [{ chunkIndex: 1, totalChunks: 1 }]
  };
}

async function splitPdfToPageImages(file: File): Promise<SplitResult> {
  if (typeof document === 'undefined') {
    throw new Error('Môi trường hiện tại không hỗ trợ render PDF bằng canvas.');
  }

  const arrayBuffer = await file.arrayBuffer();
  const loadingTask = getDocument({ data: arrayBuffer });
  const pdf = await loadingTask.promise;
  const pageCount = pdf.numPages;
  const chunks: File[] = [];
  const chunkMeta: SplitChunkMeta[] = [];

  for (let pageNumber = 1; pageNumber <= pageCount; pageNumber++) {
    const page = await pdf.getPage(pageNumber);
    const viewport = page.getViewport({ scale: 2 });
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d');
    if (!context) {
      throw new Error(`Không tạo được context để render trang ${pageNumber}.`);
    }

    canvas.width = Math.ceil(viewport.width);
    canvas.height = Math.ceil(viewport.height);

    await page.render({ canvasContext: context, viewport, canvas }).promise;

    const blob = await new Promise<Blob>((resolve, reject) => {
      canvas.toBlob((result) => {
        if (result) resolve(result);
        else reject(new Error(`Không thể tạo ảnh từ trang ${pageNumber}.`));
      }, 'image/png');
    });

    const imageFile = new File(
      [blob],
      `${file.name.replace(/\.pdf$/i, '')}_page${pageNumber}.png`,
      { type: 'image/png' }
    );
    chunks.push(imageFile);
    chunkMeta.push({
      chunkIndex: pageNumber,
      totalChunks: pageCount,
      pageStart: pageNumber,
      pageEnd: pageNumber
    });

    canvas.width = 0;
    canvas.height = 0;
  }

  return {
    chunks,
    mode: 'pdf-page-images',
    sourcePageCount: pageCount,
    processedPageCount: chunks.length,
    chunkMeta
  };
}

async function splitPdf(file: File, pagesPerChunk: number): Promise<SplitResult> {
  const arrayBuffer = await file.arrayBuffer();
  const pdfDoc = await PDFDocument.load(arrayBuffer);
  const pageCount = pdfDoc.getPageCount();
  
  if (pageCount <= pagesPerChunk) {
    return {
      chunks: [file],
      mode: 'pdf-chunk',
      sourcePageCount: pageCount,
      processedPageCount: pageCount,
      chunkMeta: [{ chunkIndex: 1, totalChunks: 1, pageStart: 1, pageEnd: pageCount }]
    };
  }

  const chunks: File[] = [];
  const chunkMeta: SplitChunkMeta[] = [];
  const totalChunks = Math.ceil(pageCount / pagesPerChunk);
  for (let i = 0; i < pageCount; i += pagesPerChunk) {
    const newPdf = await PDFDocument.create();
    const end = Math.min(i + pagesPerChunk, pageCount);
    const pages = await newPdf.copyPages(pdfDoc, Array.from({ length: end - i }, (_, k) => i + k));
    pages.forEach(page => newPdf.addPage(page));
    const pdfBytes = await newPdf.save();
    
    const chunkFile = new File([pdfBytes], `${file.name.replace('.pdf', '')}_part${Math.floor(i / pagesPerChunk) + 1}.pdf`, {
      type: 'application/pdf'
    });
    chunks.push(chunkFile);
    chunkMeta.push({
      chunkIndex: Math.floor(i / pagesPerChunk) + 1,
      totalChunks,
      pageStart: i + 1,
      pageEnd: end
    });
  }
  return {
    chunks,
    mode: 'pdf-chunk',
    sourcePageCount: pageCount,
    processedPageCount: pageCount,
    chunkMeta
  };
}

async function splitExcel(file: File, rowsPerChunk: number): Promise<SplitResult> {
  const arrayBuffer = await file.arrayBuffer();
  const workbook = XLSX.read(arrayBuffer, { type: 'array' });
  const sheetName = workbook.SheetNames[0];
  const worksheet = workbook.Sheets[sheetName];
  const jsonData = XLSX.utils.sheet_to_json(worksheet, { header: 1 }) as any[][];

  if (jsonData.length <= rowsPerChunk + 1) {
    return {
      chunks: [file],
      mode: 'excel-chunk',
      chunkMeta: [{ chunkIndex: 1, totalChunks: 1 }]
    };
  }

  const header = jsonData[0];
  const data = jsonData.slice(1);
  const chunks: File[] = [];
  const chunkMeta: SplitChunkMeta[] = [];
  const totalChunks = Math.ceil(data.length / rowsPerChunk);

  for (let i = 0; i < data.length; i += rowsPerChunk) {
    const chunkData = [header, ...data.slice(i, i + rowsPerChunk)];
    const newSheet = XLSX.utils.aoa_to_sheet(chunkData);
    const newWorkbook = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(newWorkbook, newSheet, sheetName);
    const excelBuffer = XLSX.write(newWorkbook, { bookType: 'xlsx', type: 'array' });
    
    const chunkFile = new File([excelBuffer], `${file.name.split('.')[0]}_part${Math.floor(i / rowsPerChunk) + 1}.xlsx`, {
      type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    });
    chunks.push(chunkFile);
    chunkMeta.push({
      chunkIndex: Math.floor(i / rowsPerChunk) + 1,
      totalChunks
    });
  }
  return {
    chunks,
    mode: 'excel-chunk',
    chunkMeta
  };
}
