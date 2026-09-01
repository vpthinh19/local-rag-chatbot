export const shouldPollDocuments = (documents) => documents.some(({ status }) => status === "processing" || status === "deleting");

export const documentActions = ({ status }) => {
  if (status === "failed") return ["retry", "delete"];
  if (status === "deleting") return [];
  return ["download", "delete"];
};

export const reduceStreamEvent = ({ buffers, sessionId, event }) => {
  const current = buffers[sessionId] || { text: "", status: "" };
  const next = { ...current };
  if (event.type === "delta") next.text += event.text || "";
  if (event.type === "status") next.status = event.text || "";
  if (event.type === "error") { next.status = "Lỗi"; next.terminal = "error"; }
  if (event.type === "cancelled") { next.status = "Đã dừng."; next.terminal = "cancelled"; }
  if (event.type === "done") next.terminal = "done";
  return { ...buffers, [sessionId]: next };
};
