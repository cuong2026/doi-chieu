import { PDFDocument } from 'pdf-lib';
import * as XLSX from 'xlsx';

export async function splitFile(file: File, maxPages: number = 10, maxRows: number = 500): Promise<File[]> {
  const type = file.type;
  const extension = file.name.split('.').pop()?.toLowerCase();

  if (type === 'application/pdf' || extension === 'pdf') {
    return splitPdf(file, maxPages);
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
  return [file];
}

async function splitPdf(file: File, pagesPerChunk: number): Promise<File[]> {
  const arrayBuffer = await file.arrayBuffer();
  const pdfDoc = await PDFDocument.load(arrayBuffer);
  const pageCount = pdfDoc.getPageCount();
  
  if (pageCount <= pagesPerChunk) return [file];

  const chunks: File[] = [];
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
  }
  return chunks;
}

async function splitExcel(file: File, rowsPerChunk: number): Promise<File[]> {
  const arrayBuffer = await file.arrayBuffer();
  const workbook = XLSX.read(arrayBuffer, { type: 'array' });
  const sheetName = workbook.SheetNames[0];
  const worksheet = workbook.Sheets[sheetName];
  const jsonData = XLSX.utils.sheet_to_json(worksheet, { header: 1 }) as any[][];

  if (jsonData.length <= rowsPerChunk + 1) return [file]; // +1 for header

  const header = jsonData[0];
  const data = jsonData.slice(1);
  const chunks: File[] = [];

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
  }
  return chunks;
}
