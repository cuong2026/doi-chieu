/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useRef } from 'react';
import { Upload, FileText, AlertCircle, CheckCircle2, XCircle, HelpCircle, ArrowRight, Download, RefreshCw, Loader2, FileImage, Filter, Copy } from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import { processDocument, processDocuments } from './lib/gemini';
import { processDocumentOpenAI, processDocumentsOpenAI } from './lib/openai';
import { generateReport } from './lib/compare';
import { DocumentData, ReportData, MatchStatus, AIProvider, CompareField, LineItem, ItemCodeLocation } from './types';
import { splitFileWithMetadata } from './lib/fileSplitter';
import ExcelJS from 'exceljs';
import { saveAs } from 'file-saver';

type AppState = 'UPLOAD' | 'PROCESSING' | 'RESULTS';

const DEFAULT_COMPARE_FIELDS: CompareField[] = ['itemName', 'itemCode', 'unit', 'quantity', 'unitPrice', 'totalPrice'];

const COMPARE_FIELD_OPTIONS: { key: CompareField; label: string; description: string }[] = [
  { key: 'itemName', label: 'Tên hàng', description: 'Báo lệch khi tên hàng khác nhau.' },
  { key: 'itemCode', label: 'Mã', description: 'Báo lệch khi mã hàng khác nhau.' },
  { key: 'unit', label: 'Đơn vị tính', description: 'Báo lệch khi đơn vị tính khác nhau.' },
  { key: 'quantity', label: 'Số lượng', description: 'Báo lệch khi số lượng khác nhau.' },
  { key: 'unitPrice', label: 'Đơn giá', description: 'Báo lệch khi đơn giá khác nhau.' },
  { key: 'totalPrice', label: 'Thành tiền', description: 'Báo lệch khi thành tiền khác nhau.' },
];

/** Số ảnh trang PDF gửi mỗi lần gọi AI (chỉ áp dụng mode render từng trang ảnh). */
const IMAGES_PER_BATCH = 4;

const COMPARE_FIELD_LABELS: Record<CompareField, string> = {
  itemName: 'Tên hàng',
  itemCode: 'Mã',
  unit: 'Đơn vị tính',
  quantity: 'Số lượng',
  unitPrice: 'Đơn giá',
  totalPrice: 'Thành tiền',
};

export default function App() {
  const [appState, setAppState] = useState<AppState>('UPLOAD');
  const [files, setFiles] = useState<File[]>([]);
  const [processingStatus, setProcessingStatus] = useState<string>('');
  const [reportData, setReportData] = useState<ReportData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [statusFilters, setStatusFilters] = useState<MatchStatus[]>(['MATCH', 'MISMATCH', 'MISSING', 'UNCERTAIN']);
  const [copySuccess, setCopySuccess] = useState<boolean>(false);
  const [selectedBaseFileName, setSelectedBaseFileName] = useState<string | null>(null);
  const [aiProvider, setAiProvider] = useState<AIProvider>('openai');
  const [compareFields, setCompareFields] = useState<CompareField[]>(DEFAULT_COMPARE_FIELDS);
  const [discrepancyFilter, setDiscrepancyFilter] = useState<'itemName' | 'unit' | 'quantity' | 'unitPrice' | 'totalPrice' | null>(null);
  const [itemCodeLocations, setItemCodeLocations] = useState<Record<string, ItemCodeLocation>>({});
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const selectedFiles = Array.from(e.target.files);
      if (selectedFiles.length < 2) {
        setError('Vui lòng chọn ít nhất 2 file để đối chiếu.');
        return;
      }
      if (selectedFiles.length > 4) {
        setError('Chỉ hỗ trợ tối đa 4 file cùng lúc.');
        return;
      }
      setError(null);
      setFiles(selectedFiles);
    }
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    if (e.dataTransfer.files) {
      const droppedFiles = Array.from(e.dataTransfer.files);
      if (droppedFiles.length < 2) {
        setError('Vui lòng chọn ít nhất 2 file để đối chiếu.');
        return;
      }
      if (droppedFiles.length > 4) {
        setError('Chỉ hỗ trợ tối đa 4 file cùng lúc.');
        return;
      }
      setError(null);
      setFiles(droppedFiles);
    }
  };

  const toggleCompareField = (field: CompareField) => {
    setCompareFields(prev => {
      if (prev.includes(field)) {
        if (prev.length === 1) return prev;
        return prev.filter(item => item !== field);
      }
      return [...prev, field];
    });
  };

  const handleProcess = async () => {
    if (files.length < 2) return;
    if (compareFields.length === 0) {
      setError('Vui lòng chọn ít nhất 1 trường thông tin để đối chiếu.');
      return;
    }

    console.log(`[DEBUG APP] Bắt đầu xử lý ${files.length} files...`);
    setAppState('PROCESSING');
    setError(null);
    const extractedDocs: DocumentData[] = [];

    try {
      for (let i = 0; i < files.length; i++) {
        const originalFile = files[i];
        console.log(`[DEBUG APP] Đang kiểm tra và cắt file ${i + 1}/${files.length}: ${originalFile.name}`);
        setProcessingStatus(`Đang kiểm tra file ${i + 1}/${files.length}: ${originalFile.name}...`);

        // Split file if needed (PDF scan: render theo từng trang ảnh để OCR chắc chắn đủ trang)
        const splitResult = await splitFileWithMetadata(originalFile, 10, 500);
        const chunks = splitResult.chunks;

        if (
          typeof splitResult.sourcePageCount === 'number' &&
          typeof splitResult.processedPageCount === 'number' &&
          splitResult.sourcePageCount !== splitResult.processedPageCount
        ) {
          throw new Error(
            `Phát hiện thiếu trang khi cắt file ${originalFile.name}: gốc ${splitResult.sourcePageCount} trang, xử lý ${splitResult.processedPageCount} trang.`
          );
        }

        if (splitResult.mode === 'pdf-page-images') {
          console.log(
            `[DEBUG SPLIT] ${originalFile.name}: PDF ${splitResult.sourcePageCount} trang, đã render ${splitResult.processedPageCount} ảnh/trang.`
          );
        } else if (splitResult.mode === 'pdf-chunk') {
          console.log(
            `[DEBUG SPLIT] ${originalFile.name}: fallback cắt PDF theo chunk (${chunks.length} phần).`
          );
        }

        const chunkResults: DocumentData[] = [];

        const runSingleChunk = async (chunk: File, chunkLabel: string, totalParts: number) => {
          const chunkStatus = totalParts > 1
            ? `Đang đọc file ${i + 1}/${files.length}: ${originalFile.name} (${chunkLabel})...`
            : `Đang đọc file ${i + 1}/${files.length}: ${originalFile.name}...`;
          setProcessingStatus(chunkStatus);
          console.log(`[DEBUG APP] ${chunkStatus}`);

          const fileItemCodeLocation = itemCodeLocations[originalFile.name] || 'auto';
          const docData = aiProvider === 'openai'
            ? await processDocumentOpenAI(chunk, originalFile.name, fileItemCodeLocation)
            : await processDocument(chunk, originalFile.name, fileItemCodeLocation);
          chunkResults.push(docData);
        };

        if (splitResult.mode === 'pdf-page-images') {
          const totalBatches = Math.ceil(chunks.length / IMAGES_PER_BATCH);
          for (let start = 0; start < chunks.length; start += IMAGES_PER_BATCH) {
            const batch = chunks.slice(start, start + IMAGES_PER_BATCH);
            const metaFirst = splitResult.chunkMeta[start];
            const metaLast = splitResult.chunkMeta[start + batch.length - 1];
            const batchIndex = Math.floor(start / IMAGES_PER_BATCH) + 1;
            const chunkLabel =
              metaFirst?.pageStart != null && metaLast?.pageEnd != null
                ? `Trang ${metaFirst.pageStart}-${metaLast.pageEnd} (lô ${batchIndex}/${totalBatches})`
                : `Lô ${batchIndex}/${totalBatches}`;
            const chunkStatus = `Đang đọc file ${i + 1}/${files.length}: ${originalFile.name} (${chunkLabel})...`;
            setProcessingStatus(chunkStatus);
            console.log(`[DEBUG APP] ${chunkStatus}`);

            const fileItemCodeLocation = itemCodeLocations[originalFile.name] || 'auto';
            const docData = aiProvider === 'openai'
              ? await processDocumentsOpenAI(batch, originalFile.name, fileItemCodeLocation)
              : await processDocuments(batch, originalFile.name, fileItemCodeLocation);
            chunkResults.push(docData);
          }
        } else {
          for (let j = 0; j < chunks.length; j++) {
            const chunk = chunks[j];
            const chunkMeta = splitResult.chunkMeta[j];
            const chunkLabel =
              chunkMeta?.pageStart && chunkMeta?.pageEnd
                ? `Trang ${chunkMeta.pageStart}${chunkMeta.pageStart !== chunkMeta.pageEnd ? `-${chunkMeta.pageEnd}` : ''}`
                : `Phần ${j + 1}/${chunks.length}`;
            await runSingleChunk(chunk, chunkLabel, chunks.length);
          }
        }

        // Merge chunk results for this file
        if (chunkResults.length > 0) {
          const mergedDoc: DocumentData = {
            fileName: originalFile.name,
            documentType: chunkResults[0].documentType,
            documentNumber: chunkResults[0].documentNumber,
            date: chunkResults[0].date,
            lineItems: chunkResults.flatMap(res => res.lineItems)
          };

          // Re-index line items after merging to ensure unique IDs and correct originalIndex
          mergedDoc.lineItems = mergedDoc.lineItems.map((item, idx) => ({
            ...item,
            id: `${originalFile.name}-item-${idx}`,
            originalIndex: idx + 1
          }));

          extractedDocs.push(mergedDoc);
        }
      }

      console.log(`[DEBUG APP] Toàn bộ dữ liệu đã trích xuất:`, extractedDocs);
      setProcessingStatus('Đang phân tích và đối chiếu dữ liệu (Ngữ nghĩa)...');
      const report = await generateReport(extractedDocs, selectedBaseFileName, aiProvider, compareFields);
      console.log(`[DEBUG APP] Báo cáo đối chiếu:`, report);

      setReportData(report);
      setAppState('RESULTS');
    } catch (err: any) {
      console.error(err);
      setError(err.message || 'Đã xảy ra lỗi trong quá trình xử lý.');
      setAppState('UPLOAD');
    }
  };

  const resetApp = () => {
    setFiles([]);
    setReportData(null);
    setError(null);
    setStatusFilters(['MATCH', 'MISMATCH', 'MISSING', 'UNCERTAIN']);
    setSelectedBaseFileName(null);
    setCompareFields(DEFAULT_COMPARE_FIELDS);
    setItemCodeLocations({});
    setAppState('UPLOAD');
  };

  const getStatusIcon = (status: MatchStatus) => {
    switch (status) {
      case 'MATCH': return <CheckCircle2 className="w-5 h-5 text-emerald-500" />;
      case 'MISMATCH': return <XCircle className="w-5 h-5 text-rose-500" />;
      case 'MISSING': return <AlertCircle className="w-5 h-5 text-slate-400" />;
      case 'UNCERTAIN': return <HelpCircle className="w-5 h-5 text-amber-500" />;
    }
  };

  const getStatusBadge = (status: MatchStatus) => {
    switch (status) {
      case 'MATCH': return <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-emerald-100 text-emerald-700">Khớp</span>;
      case 'MISMATCH': return <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-rose-100 text-rose-700">Lệch</span>;
      case 'MISSING': return <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-slate-100 text-slate-700">Thiếu</span>;
      case 'UNCERTAIN': return <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-amber-100 text-amber-700">Nghi ngờ</span>;
    }
  };

  const exportExcel = async () => {
    if (!reportData) return;

    const workbook = new ExcelJS.Workbook();
    const worksheet = workbook.addWorksheet('Báo cáo đối chiếu');

    // Helper function to get field value from item
    const getFieldValue = (item: LineItem, field: CompareField): any => {
      switch (field) {
        case 'itemName': return item.itemName;
        case 'itemCode': return item.itemCode || '';
        case 'unit': return item.unit || '';
        case 'quantity': return item.quantity ?? '';
        case 'unitPrice': return item.unitPrice ?? '';
        case 'totalPrice': return item.totalPrice ?? '';
        default: return '';
      }
    };

    // Build headers
    const headers: string[] = [];
    reportData.compareFields.forEach(field => {
      headers.push(`${COMPARE_FIELD_LABELS[field]} (Gốc)`);
    });

    reportData.otherFiles.forEach(f => {
      headers.push(`Trạng thái (${f.fileName})`);
      reportData.compareFields.forEach(field => {
        headers.push(`${COMPARE_FIELD_LABELS[field]} (${f.fileName})`);
      });
      headers.push(`Chi tiết lệch (${f.fileName})`);
    });

    const headerRow = worksheet.addRow(headers);

    // Style header
    headerRow.eachCell((cell) => {
      cell.font = { bold: true, color: { argb: 'FFFFFFFF' } };
      cell.fill = {
        type: 'pattern',
        pattern: 'solid',
        fgColor: { argb: 'FF4F81BD' }
      };
      cell.alignment = { vertical: 'middle', horizontal: 'center', wrapText: true };
      cell.border = {
        top: { style: 'thin', color: { argb: 'FFBFBFBF' } },
        left: { style: 'thin', color: { argb: 'FFBFBFBF' } },
        bottom: { style: 'thin', color: { argb: 'FFBFBFBF' } },
        right: { style: 'thin', color: { argb: 'FFBFBFBF' } }
      };
    });

    // Rows
    reportData.results.forEach(result => {
      const rowData: any[] = [];

      // Add base item fields
      reportData.compareFields.forEach(field => {
        rowData.push(getFieldValue(result.baseItem, field));
      });

      reportData.otherFiles.forEach(f => {
        const comp = result.comparisons[f.fileName];
        let statusText = '';
        switch (comp.status) {
          case 'MATCH': statusText = 'Khớp'; break;
          case 'MISMATCH': statusText = 'Lệch'; break;
          case 'MISSING': statusText = 'Thiếu'; break;
          case 'UNCERTAIN': statusText = 'Nghi ngờ'; break;
        }
        rowData.push(statusText);

        // Add matched item fields
        reportData.compareFields.forEach(field => {
          rowData.push(comp.matchedItem ? getFieldValue(comp.matchedItem, field) : '');
        });

        rowData.push(comp.discrepancies.join('; '));
      });

      const row = worksheet.addRow(rowData);

      // Add borders to all cells in row
      row.eachCell((cell) => {
        cell.border = {
          top: { style: 'thin', color: { argb: 'FFBFBFBF' } },
          left: { style: 'thin', color: { argb: 'FFBFBFBF' } },
          bottom: { style: 'thin', color: { argb: 'FFBFBFBF' } },
          right: { style: 'thin', color: { argb: 'FFBFBFBF' } }
        };
        cell.alignment = { vertical: 'middle', wrapText: true };
      });

      // Apply conditional formatting
      const baseColCount = reportData.compareFields.length;
      const getFieldColIndexInOther = (fileIndex: number, field: CompareField): number => {
        // Format: compareField cols + status + compareField cols
        const statusColInFile = baseColCount + 1 + fileIndex * (baseColCount + 2);
        const fieldIndexInCompareFields = reportData.compareFields.indexOf(field);
        return statusColInFile + 1 + fieldIndexInCompareFields;
      };
      const getBaseFieldColIndex = (field: CompareField): number => {
        return reportData.compareFields.indexOf(field) + 1; // 1-indexed
      };

      let baseNameHighlight: 'RED' | 'YELLOW' | null = null;
      let baseQuantityHighlight: 'RED' | 'YELLOW' | null = null;
      let basePriceHighlight: 'RED' | 'YELLOW' | null = null;

      reportData.otherFiles.forEach((f, index) => {
        const comp = result.comparisons[f.fileName];
        const statusColIndex = baseColCount + 1 + index * (baseColCount + 2); // 1-indexed

        // Status column
        const statusCell = row.getCell(statusColIndex);
        statusCell.alignment = { vertical: 'middle', horizontal: 'center', wrapText: true };

        if (comp.status === 'MATCH') {
          statusCell.font = { color: { argb: 'FF00B050' }, bold: true }; // Green
        } else if (comp.status === 'MISMATCH') {
          statusCell.font = { color: { argb: 'FFC00000' }, bold: true }; // Dark Red
          statusCell.fill = { type: 'pattern', pattern: 'solid', fgColor: { argb: 'FFFFC7CE' } }; // Light Red BG
        } else if (comp.status === 'MISSING') {
          statusCell.font = { color: { argb: 'FF7B7B7B' }, bold: true }; // Gray
          statusCell.fill = { type: 'pattern', pattern: 'solid', fgColor: { argb: 'FFF2F2F2' } }; // Light Gray BG

          // Highlight the whole block for missing item
          for (let i = 1; i <= baseColCount + 1; i++) {
            row.getCell(statusColIndex + i).fill = { type: 'pattern', pattern: 'solid', fgColor: { argb: 'FFF2F2F2' } };
          }
        } else if (comp.status === 'UNCERTAIN') {
          statusCell.font = { color: { argb: 'FF9C6500' }, bold: true }; // Dark Orange
          statusCell.fill = { type: 'pattern', pattern: 'solid', fgColor: { argb: 'FFFFEB9C' } }; // Light Orange BG
        }

        // Highlight discrepancies
        if (comp.status === 'MISMATCH' || comp.status === 'UNCERTAIN') {
          const isMismatch = comp.status === 'MISMATCH';
          const highlightColor = isMismatch ? 'FFFFC7CE' : 'FFFFEB9C'; // Light Red : Light Yellow
          const fontColor = isMismatch ? 'FFC00000' : 'FF9C6500'; // Dark Red : Dark Yellow

          // Check specific fields if they mismatch and are selected
          if (comp.matchedItem) {
            if (reportData.compareFields.includes('itemName') && comp.matchedItem.itemName !== result.baseItem.itemName) {
              const colIdx = getFieldColIndexInOther(index, 'itemName');
              row.getCell(colIdx).fill = { type: 'pattern', pattern: 'solid', fgColor: { argb: highlightColor } };
              row.getCell(colIdx).font = { color: { argb: fontColor } };
              if (isMismatch) baseNameHighlight = 'RED';
              else if (!baseNameHighlight) baseNameHighlight = 'YELLOW';
            }
            if (reportData.compareFields.includes('itemCode') && comp.matchedItem.itemCode !== result.baseItem.itemCode) {
              const colIdx = getFieldColIndexInOther(index, 'itemCode');
              row.getCell(colIdx).fill = { type: 'pattern', pattern: 'solid', fgColor: { argb: highlightColor } };
              row.getCell(colIdx).font = { color: { argb: fontColor } };
            }
            if (reportData.compareFields.includes('quantity') && comp.matchedItem.quantity !== result.baseItem.quantity) {
              const colIdx = getFieldColIndexInOther(index, 'quantity');
              row.getCell(colIdx).fill = { type: 'pattern', pattern: 'solid', fgColor: { argb: highlightColor } };
              row.getCell(colIdx).font = { color: { argb: fontColor } };
              if (isMismatch) baseQuantityHighlight = 'RED';
              else if (!baseQuantityHighlight) baseQuantityHighlight = 'YELLOW';
            }
            if (reportData.compareFields.includes('unitPrice') && comp.matchedItem.unitPrice !== result.baseItem.unitPrice) {
              const colIdx = getFieldColIndexInOther(index, 'unitPrice');
              row.getCell(colIdx).fill = { type: 'pattern', pattern: 'solid', fgColor: { argb: highlightColor } };
              row.getCell(colIdx).font = { color: { argb: fontColor } };
              if (isMismatch) basePriceHighlight = 'RED';
              else if (!basePriceHighlight) basePriceHighlight = 'YELLOW';
            }
          }

          // Discrepancies cell
          const discrepanciesColIdx = statusColIndex + baseColCount + 1;
          row.getCell(discrepanciesColIdx).font = { color: { argb: fontColor } };
        }
      });

      // Apply highlights to base columns
      const applyBaseHighlight = (field: CompareField, highlightType: 'RED' | 'YELLOW' | null) => {
        if (reportData.compareFields.includes(field)) {
          const colIndex = getBaseFieldColIndex(field);
          if (highlightType === 'RED') {
            row.getCell(colIndex).fill = { type: 'pattern', pattern: 'solid', fgColor: { argb: 'FFFFC7CE' } };
            row.getCell(colIndex).font = { color: { argb: 'FFC00000' } };
          } else if (highlightType === 'YELLOW') {
            row.getCell(colIndex).fill = { type: 'pattern', pattern: 'solid', fgColor: { argb: 'FFFFEB9C' } };
            row.getCell(colIndex).font = { color: { argb: 'FF9C6500' } };
          }
        }
      };

      applyBaseHighlight('itemName', baseNameHighlight);
      applyBaseHighlight('quantity', baseQuantityHighlight);
      applyBaseHighlight('unitPrice', basePriceHighlight);
    });

    // Auto-fit columns
    worksheet.columns.forEach((column) => {
      let maxLength = 0;
      column.eachCell!({ includeEmpty: true }, cell => {
        const columnLength = cell.value ? cell.value.toString().length : 10;
        if (columnLength > maxLength) {
          maxLength = columnLength;
        }
      });
      // Set width with some padding, max 50
      column.width = Math.min(Math.max(maxLength + 2, 12), 50);
    });

    // Generate and save file
    const buffer = await workbook.xlsx.writeBuffer();
    const blob = new Blob([buffer], { type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' });
    saveAs(blob, 'bao_cao_doi_chieu.xlsx');
  };

  const copyForGoogleSheets = () => {
    if (!reportData) return;

    // Helper function to get field value from item
    const getFieldValue = (item: LineItem, field: CompareField): any => {
      switch (field) {
        case 'itemName': return item.itemName;
        case 'itemCode': return item.itemCode || '';
        case 'unit': return item.unit || '';
        case 'quantity': return item.quantity ?? '';
        case 'unitPrice': return item.unitPrice ?? '';
        case 'totalPrice': return item.totalPrice ?? '';
        default: return '';
      }
    };

    // Build headers
    const headers: string[] = [];
    reportData.compareFields.forEach(field => {
      headers.push(`${COMPARE_FIELD_LABELS[field]} (Gốc)`);
    });

    reportData.otherFiles.forEach(f => {
      headers.push(`Trạng thái (${f.fileName})`);
      reportData.compareFields.forEach(field => {
        headers.push(`${COMPARE_FIELD_LABELS[field]} (${f.fileName})`);
      });
      headers.push(`Chi tiết lệch (${f.fileName})`);
    });

    // Rows
    const rows = reportData.results.map(result => {
      const row: any[] = [];

      // Add base item fields
      reportData.compareFields.forEach(field => {
        row.push(getFieldValue(result.baseItem, field));
      });

      reportData.otherFiles.forEach(f => {
        const comp = result.comparisons[f.fileName];
        row.push(comp.status);

        // Add matched item fields
        reportData.compareFields.forEach(field => {
          row.push(comp.matchedItem ? getFieldValue(comp.matchedItem, field) : '');
        });

        row.push(comp.discrepancies.join('; '));
      });

      return row.map(cell => {
        const cellStr = String(cell);
        // Escape tabs and newlines for TSV
        return cellStr.replace(/\t/g, ' ').replace(/\n/g, ' ');
      }).join('\t');
    });

    const tsvContent = [headers.join('\t'), ...rows].join('\n');

    navigator.clipboard.writeText(tsvContent).then(() => {
      setCopySuccess(true);
      setTimeout(() => setCopySuccess(false), 3000);
    }).catch(err => {
      console.error('Failed to copy: ', err);
      setError('Không thể copy dữ liệu. Vui lòng thử lại.');
    });
  };

  const toggleFilter = (status: MatchStatus) => {
    setStatusFilters(prev => {
      if (prev.includes(status)) {
        return prev.filter(s => s !== status);
      } else {
        return [...prev, status];
      }
    });
  };

  const toggleAllFilters = () => {
    if (statusFilters.length === 4) {
      setStatusFilters([]);
    } else {
      setStatusFilters(['MATCH', 'MISMATCH', 'MISSING', 'UNCERTAIN']);
    }
  };

  const statusCounts = {
    MATCH: 0,
    MISMATCH: 0,
    MISSING: 0,
    UNCERTAIN: 0
  };

  if (reportData) {
    reportData.results.forEach(result => {
      const statuses = Object.values(result.comparisons).map((c: any) => c.status);
      let overallStatus: MatchStatus = 'MATCH';
      if (statuses.includes('MISSING')) overallStatus = 'MISSING';
      else if (statuses.includes('MISMATCH')) overallStatus = 'MISMATCH';
      else if (statuses.includes('UNCERTAIN')) overallStatus = 'UNCERTAIN';

      statusCounts[overallStatus]++;
    });
  }

  const filteredResults = reportData?.results.filter(result => {
    // 1. Status Filter
    let matchesStatus = true;
    if (statusFilters.length === 0) {
      matchesStatus = false;
    } else if (statusFilters.length < 4) {
      const statuses = Object.values(result.comparisons).map((c: any) => c.status);
      let overallStatus: MatchStatus = 'MATCH';
      if (statuses.includes('MISSING')) overallStatus = 'MISSING';
      else if (statuses.includes('MISMATCH')) overallStatus = 'MISMATCH';
      else if (statuses.includes('UNCERTAIN')) overallStatus = 'UNCERTAIN';
      matchesStatus = statusFilters.includes(overallStatus);
    }
    
    // 2. Discrepancy Field Filter
    let matchesDiscrepancy = true;
    if (discrepancyFilter) {
      matchesDiscrepancy = Object.values(result.comparisons).some((comp: any) => {
        if (!comp.matchedItem) return false;
        if (discrepancyFilter === 'unit') return comp.matchedItem.unit !== result.baseItem.unit;
        if (discrepancyFilter === 'quantity') return comp.matchedItem.quantity !== result.baseItem.quantity;
        if (discrepancyFilter === 'unitPrice') return comp.matchedItem.unitPrice !== result.baseItem.unitPrice;
        if (discrepancyFilter === 'totalPrice') return comp.matchedItem.totalPrice !== result.baseItem.totalPrice;
        if (discrepancyFilter === 'itemName') return comp.matchedItem.itemName !== result.baseItem.itemName;
        return false;
      });
    }

    return matchesStatus && matchesDiscrepancy;
  }) || [];

  return (
    <div className="min-h-screen font-sans flex flex-col">
      <header className="bg-white border-b border-slate-200 sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="bg-blue-600 p-2 rounded-lg">
              <FileText className="w-5 h-5 text-white" />
            </div>
            <h1 className="text-xl font-bold text-slate-900 tracking-tight">Đối chiếu chứng từ tự động</h1>
          </div>
          {appState === 'RESULTS' && (
            <button
              onClick={resetApp}
              className="flex items-center gap-2 px-4 py-2 text-sm font-medium text-slate-600 hover:text-slate-900 hover:bg-slate-100 rounded-lg transition-colors"
            >
              <RefreshCw className="w-4 h-4" />
              Kiểm tra đơn mới
            </button>
          )}
        </div>
      </header>

      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <AnimatePresence mode="wait">
          {appState === 'UPLOAD' && (
            <motion.div
              key="upload"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="max-w-3xl mx-auto"
            >
              <div className="text-center mb-8">
                <h2 className="text-3xl font-bold text-slate-900 mb-4">Đối chiếu chứng từ tự động</h2>
                <p className="text-slate-600 text-lg">
                  Tải lên 2 đến 4 chứng từ của cùng một đơn hàng (Đơn đặt hàng, Phiếu xuất kho, Hóa đơn...).
                  Hệ thống sẽ tự động nhận diện và tìm ra các điểm sai lệch.
                </p>
              </div>

              <div
                className="border-2 border-dashed border-slate-300 rounded-2xl bg-white p-12 text-center hover:border-blue-500 hover:bg-blue-50 transition-all cursor-pointer group"
                onDragOver={(e) => e.preventDefault()}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
              >
                <input
                  type="file"
                  multiple
                  accept="image/*,application/pdf"
                  className="hidden"
                  ref={fileInputRef}
                  onChange={handleFileSelect}
                />
                <div className="bg-blue-100 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-6 group-hover:scale-110 transition-transform">
                  <Upload className="w-8 h-8 text-blue-600" />
                </div>
                <h3 className="text-xl font-semibold text-slate-900 mb-2">Kéo thả file vào đây</h3>
                <p className="text-slate-500 mb-6">hoặc click để chọn file từ máy tính (Hỗ trợ PDF, JPG, PNG)</p>

                {files.length > 0 && (
                  <div className="mt-8 text-left bg-slate-50 rounded-xl p-4 border border-slate-200" onClick={(e) => e.stopPropagation()}>
                    <h4 className="font-medium text-slate-900 mb-3 flex items-center justify-between">
                      <span>Đã chọn {files.length} file</span>
                      <button onClick={() => { setFiles([]); setSelectedBaseFileName(null); }} className="text-sm text-rose-600 hover:text-rose-700">Xóa tất cả</button>
                    </h4>
                    <ul className="space-y-3 mb-4">
                      {files.map((f, i) => (
                        <li key={i} className="flex flex-col gap-2 text-sm text-slate-600 bg-white p-3 rounded-lg border border-slate-200 shadow-sm">
                          <div className="flex items-center gap-3">
                            <FileImage className="w-4 h-4 text-blue-500 shrink-0" />
                            <span className="truncate flex-1 font-medium text-slate-800">{f.name}</span>
                            <span className="text-xs text-slate-400 shrink-0">{(f.size / 1024 / 1024).toFixed(2)} MB</span>
                          </div>
                          <div className="mt-1 bg-slate-50 p-2 rounded border border-slate-100">
                            <label className="block text-xs font-semibold text-slate-600 mb-1.5 uppercase tracking-wider">
                              Khu vực trích xuất mã sản phẩm:
                            </label>
                            <select
                              value={itemCodeLocations[f.name] || 'auto'}
                              onChange={(e) => setItemCodeLocations(prev => ({ ...prev, [f.name]: e.target.value as ItemCodeLocation }))}
                              className="block w-full rounded-md border-slate-300 border py-1.5 px-2 text-xs focus:border-blue-500 focus:ring-blue-500 bg-white font-medium text-slate-700"
                              onClick={(e) => e.stopPropagation()}
                            >
                              <option value="auto">Tự động (Ưu tiên cột mã riêng, nếu không có tìm trong tên)</option>
                              <option value="separate_column">Chỉ ở cột mã riêng (Bỏ qua nếu mã nằm trong tên)</option>
                              <option value="in_name">Nằm lẫn trong tên sản phẩm</option>
                            </select>
                          </div>
                        </li>
                      ))}
                    </ul>

                    {files.length >= 2 && (
                      <div className="pt-4 border-t border-slate-200">
                        <label htmlFor="aiProviderSelect" className="block text-sm font-medium text-slate-700 mb-2 mt-4">
                          Chọn mô hình AI xử lý:
                        </label>
                        <select
                          id="aiProviderSelect"
                          value={aiProvider}
                          onChange={(e) => setAiProvider(e.target.value as AIProvider)}
                          className="block w-full rounded-lg border-slate-300 border p-2.5 text-sm focus:border-blue-500 focus:ring-blue-500 bg-white mb-4"
                        >
                          <option value="gemini">Google Gemini (gemini-3.1-flash)</option>
                          <option value="openai">OpenAI (gpt-5.4-mini)</option>
                        </select>

                        <label htmlFor="baseFileSelect" className="block text-sm font-medium text-slate-700 mb-2">
                          Chọn file gốc để đối chiếu (Tùy chọn):
                        </label>
                        <select
                          id="baseFileSelect"
                          value={selectedBaseFileName || ''}
                          onChange={(e) => setSelectedBaseFileName(e.target.value || null)}
                          className="block w-full rounded-lg border-slate-300 border p-2.5 text-sm focus:border-blue-500 focus:ring-blue-500 bg-white"
                        >
                          <option value="">Tự động chọn file có nhiều mặt hàng nhất</option>
                          {files.map((f, i) => (
                            <option key={i} value={f.name}>{f.name}</option>
                          ))}
                        </select>
                        <p className="mt-1.5 text-xs text-slate-500">
                          File gốc sẽ được dùng làm chuẩn để so sánh với các file còn lại.
                        </p>

                        <div className="mt-4">
                          <div className="flex items-center justify-between gap-3 mb-2">
                            <label className="block text-sm font-medium text-slate-700">
                              Chọn trường thông tin cần đối chiếu:
                            </label>
                            <button
                              type="button"
                              onClick={() => setCompareFields(DEFAULT_COMPARE_FIELDS)}
                              className="text-xs font-medium text-blue-600 hover:text-blue-700"
                            >
                              Chọn mặc định
                            </button>
                          </div>

                          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                            {COMPARE_FIELD_OPTIONS.map((option) => {
                              const isSelected = compareFields.includes(option.key);
                              return (
                                <button
                                  key={option.key}
                                  type="button"
                                  onClick={() => toggleCompareField(option.key)}
                                  className={`text-left rounded-xl border p-3 transition-all ${isSelected
                                      ? 'border-blue-500 bg-blue-50 shadow-sm'
                                      : 'border-slate-200 bg-white hover:border-slate-300'
                                    }`}
                                >
                                  <div className="flex items-center justify-between gap-3">
                                    <div className="font-medium text-slate-900">{option.label}</div>
                                    <div className={`w-4 h-4 rounded border flex items-center justify-center ${isSelected ? 'border-blue-600 bg-blue-600' : 'border-slate-300 bg-white'
                                      }`}>
                                      {isSelected && <CheckCircle2 className="w-3 h-3 text-white" />}
                                    </div>
                                  </div>
                                  <div className="text-xs text-slate-500 mt-1">{option.description}</div>
                                </button>
                              );
                            })}
                          </div>

                          <p className="mt-2 text-xs text-slate-500">
                            Tên hàng vẫn luôn được dùng để ghép đúng mặt hàng. Các mục bạn chọn ở đây sẽ quyết định trường nào bị báo lệch trong kết quả và file Excel.
                          </p>
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>

              {error && (
                <div className="mt-6 p-4 bg-rose-50 border border-rose-200 rounded-xl flex items-start gap-3 text-rose-700">
                  <AlertCircle className="w-5 h-5 shrink-0 mt-0.5" />
                  <p>{error}</p>
                </div>
              )}

              <div className="mt-8 flex justify-center">
                <button
                  onClick={handleProcess}
                  disabled={files.length < 2}
                  className="bg-blue-600 hover:bg-blue-700 disabled:bg-slate-300 disabled:cursor-not-allowed text-white px-8 py-3 rounded-xl font-medium text-lg flex items-center gap-2 transition-colors shadow-sm"
                >
                  Bắt đầu đối chiếu
                  <ArrowRight className="w-5 h-5" />
                </button>
              </div>
            </motion.div>
          )}

          {appState === 'PROCESSING' && (
            <motion.div
              key="processing"
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              className="max-w-md mx-auto mt-20 text-center"
            >
              <div className="relative w-24 h-24 mx-auto mb-8">
                <div className="absolute inset-0 border-4 border-slate-100 rounded-full"></div>
                <div className="absolute inset-0 border-4 border-blue-600 rounded-full border-t-transparent animate-spin"></div>
                <FileText className="absolute inset-0 m-auto w-8 h-8 text-blue-600" />
              </div>
              <h2 className="text-2xl font-bold text-slate-900 mb-2">Hệ thống đang xử lý</h2>
              <p className="text-slate-600">{processingStatus}</p>
              <p className="text-sm text-slate-400 mt-4">Quá trình này có thể mất vài chục giây tùy thuộc vào số lượng và độ phức tạp của chứng từ.</p>
            </motion.div>
          )}

          {appState === 'RESULTS' && reportData && (
            <motion.div
              key="results"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="space-y-8"
            >
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                <div>
                  <h2 className="text-2xl font-bold text-slate-900">Báo cáo đối chiếu</h2>
                  <p className="text-slate-600 mt-1">
                    Đã chọn <span className="font-semibold text-slate-900">{reportData.baseFile.fileName}</span> làm file gốc ({reportData.baseFile.lineItems.length} dòng).
                  </p>
                </div>
                <div className="flex items-center gap-2">
                  <button
                    onClick={copyForGoogleSheets}
                    className="flex items-center gap-2 bg-white border border-slate-300 hover:bg-slate-50 text-slate-700 px-4 py-2 rounded-lg font-medium transition-colors shadow-sm relative"
                  >
                    {copySuccess ? <CheckCircle2 className="w-4 h-4 text-emerald-600" /> : <Copy className="w-4 h-4" />}
                    {copySuccess ? 'Đã copy!' : 'Copy cho Google Sheets'}
                  </button>
                  <button
                    onClick={exportExcel}
                    className="flex items-center gap-2 bg-white border border-slate-300 hover:bg-slate-50 text-slate-700 px-4 py-2 rounded-lg font-medium transition-colors shadow-sm"
                  >
                    <Download className="w-4 h-4" />
                    Xuất Excel
                  </button>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm">
                  <div className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-1">File gốc</div>
                  <div className="font-medium text-slate-900 truncate" title={reportData.baseFile.fileName}>{reportData.baseFile.fileName}</div>
                  <div className="text-sm text-slate-500 mt-1">{reportData.baseFile.documentType} - {reportData.baseFile.documentNumber}</div>
                </div>
                {reportData.otherFiles.map((f, i) => (
                  <div key={i} className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm">
                    <div className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-1">File đối chiếu {i + 1}</div>
                    <div className="font-medium text-slate-900 truncate" title={f.fileName}>{f.fileName}</div>
                    <div className="text-sm text-slate-500 mt-1">{f.documentType} - {f.documentNumber}</div>
                  </div>
                ))}
              </div>

              <div className="space-y-3 bg-white p-4 rounded-xl border border-slate-200 shadow-sm">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="text-sm font-medium text-slate-500 flex items-center gap-1 mr-2">
                    <Filter className="w-4 h-4" /> Lọc trạng thái:
                  </span>
                  <button onClick={toggleAllFilters} className={`px-3 py-1.5 rounded-full text-sm font-medium transition-colors ${statusFilters.length === 4 ? 'bg-slate-800 text-white' : 'bg-slate-100 text-slate-600 hover:bg-slate-200'}`}>Tất cả ({reportData.results.length})</button>
                  <button onClick={() => toggleFilter('MISMATCH')} className={`px-3 py-1.5 rounded-full text-sm font-medium transition-colors ${statusFilters.includes('MISMATCH') ? 'bg-rose-600 text-white' : 'bg-rose-100 text-rose-700 hover:bg-rose-200'}`}>Lệch ({statusCounts.MISMATCH})</button>
                  <button onClick={() => toggleFilter('MISSING')} className={`px-3 py-1.5 rounded-full text-sm font-medium transition-colors ${statusFilters.includes('MISSING') ? 'bg-slate-600 text-white' : 'bg-slate-100 text-slate-700 hover:bg-slate-200'}`}>Thiếu ({statusCounts.MISSING})</button>
                  <button onClick={() => toggleFilter('UNCERTAIN')} className={`px-3 py-1.5 rounded-full text-sm font-medium transition-colors ${statusFilters.includes('UNCERTAIN') ? 'bg-amber-500 text-white' : 'bg-amber-100 text-amber-700 hover:bg-amber-200'}`}>Nghi ngờ ({statusCounts.UNCERTAIN})</button>
                  <button onClick={() => toggleFilter('MATCH')} className={`px-3 py-1.5 rounded-full text-sm font-medium transition-colors ${statusFilters.includes('MATCH') ? 'bg-emerald-600 text-white' : 'bg-emerald-100 text-emerald-700 hover:bg-emerald-200'}`}>Khớp hoàn toàn ({statusCounts.MATCH})</button>
                </div>

                <div className="flex flex-wrap items-center gap-2 pt-3 border-t border-slate-100">
                  <span className="text-sm font-medium text-slate-500 flex items-center gap-1 mr-2">
                    Lọc chi tiết lệch:
                  </span>
                  <button 
                    onClick={() => setDiscrepancyFilter(discrepancyFilter === 'itemName' ? null : 'itemName')} 
                    className={`px-3 py-1.5 rounded-full text-sm font-medium transition-colors ${discrepancyFilter === 'itemName' ? 'bg-blue-600 text-white' : 'bg-blue-50 text-blue-700 hover:bg-blue-100'}`}
                  >
                    Tên hàng
                  </button>
                  <button 
                    onClick={() => setDiscrepancyFilter(discrepancyFilter === 'totalPrice' ? null : 'totalPrice')} 
                    className={`px-3 py-1.5 rounded-full text-sm font-medium transition-colors ${discrepancyFilter === 'totalPrice' ? 'bg-blue-600 text-white' : 'bg-blue-50 text-blue-700 hover:bg-blue-100'}`}
                  >
                    Thành tiền
                  </button>
                  <button 
                    onClick={() => setDiscrepancyFilter(discrepancyFilter === 'unitPrice' ? null : 'unitPrice')} 
                    className={`px-3 py-1.5 rounded-full text-sm font-medium transition-colors ${discrepancyFilter === 'unitPrice' ? 'bg-blue-600 text-white' : 'bg-blue-50 text-blue-700 hover:bg-blue-100'}`}
                  >
                    Đơn giá
                  </button>
                  <button 
                    onClick={() => setDiscrepancyFilter(discrepancyFilter === 'quantity' ? null : 'quantity')} 
                    className={`px-3 py-1.5 rounded-full text-sm font-medium transition-colors ${discrepancyFilter === 'quantity' ? 'bg-blue-600 text-white' : 'bg-blue-50 text-blue-700 hover:bg-blue-100'}`}
                  >
                    Số lượng
                  </button>
                  <button 
                    onClick={() => setDiscrepancyFilter(discrepancyFilter === 'unit' ? null : 'unit')} 
                    className={`px-3 py-1.5 rounded-full text-sm font-medium transition-colors ${discrepancyFilter === 'unit' ? 'bg-blue-600 text-white' : 'bg-blue-50 text-blue-700 hover:bg-blue-100'}`}
                  >
                    Đơn vị tính
                  </button>
                </div>
              </div>

              <div className="bg-white border border-slate-200 rounded-2xl shadow-sm overflow-hidden">
                <div className="overflow-x-auto">
                  <table className="w-full text-left border-collapse">
                    <thead>
                      <tr className="bg-slate-50 border-b border-slate-200">
                        <th className="p-4 font-semibold text-slate-900 w-1/3 min-w-[300px]">
                          Thông tin từ File Gốc
                        </th>
                        {reportData.otherFiles.map((f, i) => (
                          <th key={i} className="p-4 font-semibold text-slate-900 min-w-[350px] border-l border-slate-200">
                            Đối chiếu với: <span className="text-blue-600 font-medium">{f.fileName}</span>
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-200">
                      {filteredResults.length === 0 ? (
                        <tr>
                          <td colSpan={reportData.otherFiles.length + 1} className="p-8 text-center text-slate-500">
                            Không có kết quả nào phù hợp với bộ lọc hiện tại.
                          </td>
                        </tr>
                      ) : filteredResults.map((result, idx) => (
                        <tr key={idx} className="hover:bg-slate-50/50 transition-colors">
                          <td className="p-4 align-top">
                            <div className="font-medium text-slate-900 mb-2 flex items-start gap-2">
                              <span className="shrink-0 inline-flex items-center justify-center w-5 h-5 rounded bg-slate-200 text-xs font-bold text-slate-600 mt-0.5" title="Số thứ tự dòng trong file gốc">{result.baseItem.originalIndex}</span>
                              <span>
                                {result.baseItem.itemCode && reportData.compareFields.includes('itemCode') && <span className="text-blue-600 font-semibold mr-1">[{result.baseItem.itemCode}]</span>}
                                {result.baseItem.itemName}
                              </span>
                            </div>
                            <div className="space-y-2 text-sm">
                              {reportData.compareFields.includes('unit') && (
                                <div className="bg-slate-100 p-2 rounded-md">
                                  <div className="text-xs text-slate-500 mb-0.5">{COMPARE_FIELD_LABELS['unit']}</div>
                                  <div className="font-mono">{result.baseItem.unit ?? '-'}</div>
                                </div>
                              )}
                              {reportData.compareFields.includes('quantity') && (
                                <div className="bg-slate-100 p-2 rounded-md">
                                  <div className="text-xs text-slate-500 mb-0.5">{COMPARE_FIELD_LABELS['quantity']}</div>
                                  <div className="font-mono">{result.baseItem.quantity ?? '-'}</div>
                                </div>
                              )}
                              {reportData.compareFields.includes('unitPrice') && (
                                <div className="bg-slate-100 p-2 rounded-md">
                                  <div className="text-xs text-slate-500 mb-0.5">{COMPARE_FIELD_LABELS['unitPrice']}</div>
                                  <div className="font-mono">{result.baseItem.unitPrice?.toLocaleString() ?? '-'}</div>
                                </div>
                              )}
                              {reportData.compareFields.includes('totalPrice') && (
                                <div className="bg-slate-100 p-2 rounded-md">
                                  <div className="text-xs text-slate-500 mb-0.5">{COMPARE_FIELD_LABELS['totalPrice']}</div>
                                  <div className="font-mono">{result.baseItem.totalPrice?.toLocaleString() ?? '-'}</div>
                                </div>
                              )}
                            </div>
                          </td>

                          {reportData.otherFiles.map((f, i) => {
                            const comp = result.comparisons[f.fileName];
                            return (
                              <td key={i} className="p-4 align-top border-l border-slate-200">
                                <div className="flex items-start justify-between mb-2 gap-2">
                                  <div className="flex-1">
                                    {comp.matchedItem ? (
                                      <div className="font-medium text-slate-700 flex items-start gap-2">
                                        <span className="shrink-0 inline-flex items-center justify-center w-5 h-5 rounded bg-slate-200 text-xs font-bold text-slate-600 mt-0.5" title="Số thứ tự dòng trong file đối chiếu">{comp.matchedItem.originalIndex}</span>
                                        <span>
                                          {reportData.compareFields.includes('itemCode') && comp.matchedItem.itemCode && <span className="text-blue-600 font-semibold mr-1">[{comp.matchedItem.itemCode}]</span>}
                                          {comp.matchedItem.itemName}
                                        </span>
                                      </div>
                                    ) : (
                                      <div className="text-slate-400 italic">Không tìm thấy mặt hàng tương ứng</div>
                                    )}
                                  </div>
                                  <div className="shrink-0 mt-0.5">
                                    {getStatusBadge(comp.status)}
                                  </div>
                                </div>

                                {comp.matchedItem && (
                                  <div className="space-y-2 text-sm mb-3">
                                    {reportData.compareFields.includes('unit') && (
                                      <div className={`p-2 rounded-md border ${comp.matchedItem.unit !== result.baseItem.unit ? 'bg-rose-50 border-rose-200 text-rose-700' : 'bg-emerald-50 border-emerald-200 text-emerald-700'}`}>
                                        <div className="text-xs opacity-70 mb-0.5">{COMPARE_FIELD_LABELS['unit']}</div>
                                        <div className="font-mono">{comp.matchedItem.unit ?? '-'}</div>
                                      </div>
                                    )}
                                    {reportData.compareFields.includes('quantity') && (
                                      <div className={`p-2 rounded-md border ${comp.matchedItem.quantity !== result.baseItem.quantity ? 'bg-rose-50 border-rose-200 text-rose-700' : 'bg-emerald-50 border-emerald-200 text-emerald-700'}`}>
                                        <div className="text-xs opacity-70 mb-0.5">{COMPARE_FIELD_LABELS['quantity']}</div>
                                        <div className="font-mono">{comp.matchedItem.quantity ?? '-'}</div>
                                      </div>
                                    )}
                                    {reportData.compareFields.includes('unitPrice') && (
                                      <div className={`p-2 rounded-md border ${comp.matchedItem.unitPrice !== result.baseItem.unitPrice ? 'bg-rose-50 border-rose-200 text-rose-700' : 'bg-emerald-50 border-emerald-200 text-emerald-700'}`}>
                                        <div className="text-xs opacity-70 mb-0.5">{COMPARE_FIELD_LABELS['unitPrice']}</div>
                                        <div className="font-mono">{comp.matchedItem.unitPrice?.toLocaleString() ?? '-'}</div>
                                      </div>
                                    )}
                                    {reportData.compareFields.includes('totalPrice') && (
                                      <div className={`p-2 rounded-md border ${comp.matchedItem.totalPrice !== result.baseItem.totalPrice ? 'bg-rose-50 border-rose-200 text-rose-700' : 'bg-emerald-50 border-emerald-200 text-emerald-700'}`}>
                                        <div className="text-xs opacity-70 mb-0.5">{COMPARE_FIELD_LABELS['totalPrice']}</div>
                                        <div className="font-mono">{comp.matchedItem.totalPrice?.toLocaleString() ?? '-'}</div>
                                      </div>
                                    )}
                                  </div>
                                )}

                                {comp.discrepancies.length > 0 && (
                                  <div className="space-y-1 mb-3">
                                    {comp.discrepancies.map((disc, dIdx) => (
                                      <div key={dIdx} className="flex items-start gap-1.5 text-sm text-rose-700 bg-rose-100 p-2 rounded-md font-medium">
                                        <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
                                        <span>{disc}</span>
                                      </div>
                                    ))}
                                  </div>
                                )}

                                {comp.suggestions && comp.suggestions.length > 0 && (
                                  <div className="mt-3 pt-3 border-t border-slate-200">
                                    <div className="text-xs font-semibold text-slate-500 mb-2 uppercase tracking-wider">Gợi ý mặt hàng gần giống:</div>
                                    <div className="space-y-2">
                                      {comp.suggestions.map((sug, sIdx) => (
                                        <div key={sIdx} className="text-xs bg-slate-50 p-2 rounded border border-slate-100 flex flex-col gap-1">
                                          <div className="flex justify-between items-start">
                                            <span className="font-medium text-slate-700 truncate pr-2">
                                              {sug.item.itemCode && `[${sug.item.itemCode}] `}{sug.item.itemName}
                                            </span>
                                            <span className="text-blue-600 font-semibold shrink-0">{Math.round(sug.score * 100)}%</span>
                                          </div>
                                          <div className="text-slate-500 flex gap-3">
                                            <span>SL: {sug.item.quantity ?? '-'}</span>
                                            <span>Giá: {sug.item.unitPrice?.toLocaleString() ?? '-'}</span>
                                          </div>
                                        </div>
                                      ))}
                                    </div>
                                  </div>
                                )}
                              </td>
                            );
                          })}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </main>
    </div>
  );
}
