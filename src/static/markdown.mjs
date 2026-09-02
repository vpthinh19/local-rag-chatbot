const escapeHtml = (value) => value.replace(
  /[&<>"']/g,
  (character) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[character]
);

function inline(value) {
  const code = [];
  let text = escapeHtml(value).replace(/`([^`\n]+)`/g, (_, content) => {
    code.push(`<code>${content}</code>`);
    return `\u0000${code.length - 1}\u0000`;
  });
  text = text
    .replace(/\[([^\]]+)]\(([^)\s]+)\)/g, (match, label, url) =>
      /^(https?:\/\/|mailto:|#)/i.test(url)
        ? `<a href="${url}" target="_blank" rel="noopener noreferrer">${label}</a>`
        : match
    )
    .replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>")
    .replace(/__([^_\n]+)__/g, "<strong>$1</strong>")
    .replace(/~~([^~\n]+)~~/g, "<del>$1</del>")
    .replace(/(^|[^*])\*([^*\n]+)\*/g, "$1<em>$2</em>")
    .replace(/(^|[^_])_([^_\n]+)_/g, "$1<em>$2</em>");
  return text.replace(/\u0000(\d+)\u0000/g, (_, index) => code[Number(index)]);
}

const cells = (line) => line.replace(/^\||\|$/g, "").split("|").map((cell) => cell.trim());
const isBlock = (line) => /^(#{1,6}\s|```|>\s?|[-*+]\s|\d+\.\s|---+$)/.test(line);

export function renderMarkdown(value) {
  const lines = String(value ?? "").replace(/\r\n?/g, "\n").split("\n");
  const output = [];
  for (let index = 0; index < lines.length;) {
    const line = lines[index];
    if (!line.trim()) { index += 1; continue; }
    if (line.startsWith("```")) {
      const language = line.slice(3).trim().replace(/[^\w-]/g, "");
      const body = [];
      index += 1;
      while (index < lines.length && !lines[index].startsWith("```")) body.push(lines[index++]);
      if (index < lines.length) index += 1;
      output.push(`<pre><code${language ? ` class="language-${language}"` : ""}>${escapeHtml(body.join("\n"))}</code></pre>`);
      continue;
    }
    if (line.includes("|") && index + 1 < lines.length && /^\s*\|?\s*:?-{3,}/.test(lines[index + 1])) {
      const header = cells(line).map((cell) => `<th>${inline(cell)}</th>`).join("");
      index += 2;
      const rows = [];
      while (index < lines.length && lines[index].includes("|") && lines[index].trim()) {
        rows.push(`<tr>${cells(lines[index++]).map((cell) => `<td>${inline(cell)}</td>`).join("")}</tr>`);
      }
      output.push(`<div class="table-scroll"><table><thead><tr>${header}</tr></thead><tbody>${rows.join("")}</tbody></table></div>`);
      continue;
    }
    const heading = /^(#{1,6})\s+(.+)$/.exec(line);
    if (heading) { const level = heading[1].length; output.push(`<h${level}>${inline(heading[2])}</h${level}>`); index += 1; continue; }
    const list = /^([-*+]|\d+\.)\s+(.+)$/.exec(line);
    if (list) {
      const ordered = /\d/.test(list[1]);
      const items = [];
      while (index < lines.length) {
        const item = (ordered ? /^\d+\.\s+(.+)$/ : /^[-*+]\s+(.+)$/).exec(lines[index]);
        if (!item) break;
        items.push(`<li>${inline(item[1])}</li>`); index += 1;
      }
      const tag = ordered ? "ol" : "ul"; output.push(`<${tag}>${items.join("")}</${tag}>`); continue;
    }
    if (/^>\s?/.test(line)) {
      const quote = [];
      while (index < lines.length && /^>\s?/.test(lines[index])) quote.push(lines[index++].replace(/^>\s?/, ""));
      output.push(`<blockquote>${inline(quote.join("\n")).replace(/\n/g, "<br>")}</blockquote>`); continue;
    }
    if (/^---+$/.test(line.trim())) { output.push("<hr>"); index += 1; continue; }
    const paragraph = [line]; index += 1;
    while (index < lines.length && lines[index].trim() && !isBlock(lines[index])) paragraph.push(lines[index++]);
    output.push(`<p>${inline(paragraph.join("\n")).replace(/\n/g, "<br>")}</p>`);
  }
  return output.join("");
}
