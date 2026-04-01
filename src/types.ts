export type AIProvider = 'gemini' | 'openai';

export type CompareField = 'itemName' | 'itemCode' | 'unit' | 'quantity' | 'unitPrice' | 'totalPrice';

export interface LineItem {
  id: string;
  originalIndex: number;
  itemCode: string | null;
  itemName: string;
  quantity: number | null;
  unitPrice: number | null;
  totalPrice: number | null;
  unit: string | null;
}

export interface DocumentData {
  fileName: string;
  documentType: string;
  documentNumber: string;
  date: string;
  lineItems: LineItem[];
}

export type MatchStatus = 'MATCH' | 'MISMATCH' | 'MISSING' | 'UNCERTAIN';

export interface ComparisonDetail {
  status: MatchStatus;
  matchedItem?: LineItem;
  discrepancies: string[];
  suggestions?: { item: LineItem; score: number }[];
}

export interface ComparisonResult {
  baseItem: LineItem;
  comparisons: Record<string, ComparisonDetail>;
}

export interface ReportData {
  baseFile: DocumentData;
  otherFiles: DocumentData[];
  results: ComparisonResult[];
  compareFields: CompareField[];
}