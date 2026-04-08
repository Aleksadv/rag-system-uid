import re
import os
import sys
from typing import List, Dict, Any
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer
import pytesseract
from pdf2image import convert_from_path
import cv2
import numpy as np
import platform

# -----------------------------
# Заголовочное регулярное выражение
# -----------------------------
HEADER_RE = re.compile(
    r"^\s*"                    
    r"(?:(\d+(?:\.\d+)*)\.?)?"  
    r"\s*"                      
    r"(.+?)"                    
    r"(?:\s*\(([^)]+)\))?"      
    r"\s*$",
    flags=re.M
)

# -----------------------------
# Словарь замены математических символов на слова
# -----------------------------
MATH_SYMBOLS = {
    '∀': 'forall', '∃': 'exists', '⊤': 'top', '⊥': 'bottom', '∧': 'and', '∨': 'or', '¬': 'not',
    '→': 'implies', '↔': 'iff', '≡': 'equiv', '≤': 'le', '≥': 'ge', '≠': 'ne', '≈': 'approx',
    '∑': 'sum', '∏': 'prod', '∫': 'integral', '∂': 'partial', 'π': 'pi', 'σ': 'sigma', 'λ': 'lambda',
    'θ': 'theta', '±': 'pm', '∞': 'infinity', '⊆': 'subset', '⊂': 'subset', '∪': 'union', '∩': 'intersect',
    'Δ': 'Delta', '√': 'sqrt', '∈': 'in', '∉': 'notin', '∅': 'emptyset', '⇒': 'implies', '⇔': 'iff',
    '⟨': '<', '⟩': '>', '·': '*', '∘': 'o', '⊊': 'subset', '⊋': 'superset', '⋂': 'intersect',
    '⋃': 'union', '×': 'times', '⊕': 'oplus', '⊗': 'otimes', '⊙': 'odot', '⊖': 'ominus', '⊘': 'oslash',
    '⌊': 'lfloor', '⌋': 'rfloor', '⌈': 'lceil', '⌉': 'rceil', '†': 'dagger', '‡': 'ddagger', '⋅': 'cdot',
    '…': '...', '—': '--', '–': '-', '−': '-', '′': "'", '″': "''", '‴': "'''", '°': 'deg',
    'ℕ': 'N', 'ℤ': 'Z', 'ℚ': 'Q', 'ℝ': 'R', 'ℂ': 'C', 'ℍ': 'H', 'ℙ': 'P', 'ℳ': 'M',
    'ℒ': 'L', 'ℱ': 'F', '𝒜': 'A', 'ℬ': 'B', '𝒞': 'C', '𝒟': 'D', 'ℰ': 'E', 'ℱ': 'F',
    '𝒢': 'G', 'ℋ': 'H', 'ℐ': 'I', '𝒥': 'J', '𝒦': 'K', 'ℒ': 'L', 'ℳ': 'M', '𝒩': 'N',
    '𝒪': 'O', '𝒫': 'P', '𝒬': 'Q', 'ℛ': 'R', '𝒮': 'S', '𝒯': 'T', '𝒰': 'U', '𝒱': 'V',
    '𝒲': 'W', '𝒳': 'X', '𝒴': 'Y', '𝒵': 'Z',
}
DEFAULT_FORMULA_TOKEN = " [FORMULA] "

# -----------------------------
# 0. вспомогательные функции для OCR
# -----------------------------
def preprocess_image(img):
    img = np.array(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    return thresh

def get_poppler_path():
    system = platform.system()
    if system == "Darwin":  # macOS
        return "/usr/local/bin"
    elif system == "Windows":
        return r"C:\poppler\bin"
    else:
        return None  # Linux — обычно в PATH

def pdf_to_text_ocr(pdf_path: str) -> str:
    poppler_path = get_poppler_path()
    pages = convert_from_path(pdf_path, dpi=300, poppler_path=poppler_path)
    all_text = []
    for i, page in enumerate(pages):
        print(f"[OCR] Page {i+1}")
        img = preprocess_image(page)
        text = pytesseract.image_to_string(img, lang='rus+eng', config='--oem 3 --psm 6')
        all_text.append(text)
    return "\n".join(all_text)

# -----------------------------
# 1. извлечение текста страницы
# -----------------------------
def page_to_text(page_layout) -> str:
    elements = [(el.y1, el.get_text()) for el in page_layout if isinstance(el, LTTextContainer)]
    elements.sort(key=lambda x: -x[0])
    parts = [text.replace("\x0c", "").rstrip() for _, text in elements if text.strip()]
    return "\n".join(parts)

# -----------------------------
# 2. замена математических символов на слова
# -----------------------------
def replace_math_symbols(text: str) -> str:
    for sym, word in MATH_SYMBOLS.items():
        text = re.sub(re.escape(sym), f' {word} ', text)
    return text

# -----------------------------
# 3. очистка текста
# -----------------------------
def cleaning_text(
    text: str,
    *,
    remove_latin: bool = False,
    remove_digits: bool = False,
    replace_formulas_with_token: bool = False,
    formula_token: str = DEFAULT_FORMULA_TOKEN,
    aggressive_math_removal: bool = False,
    remove_math_symbols: bool = False,
    replace_symbols_with_words: bool = True,
) -> str:
    if not text:
        return ""

    # --- Замена математических символов на слова ---
    if replace_symbols_with_words:
        text = replace_math_symbols(text)

    # --- Нормализация переносов и лишних символов ---
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\(cid:\d+\)", " ", text)
    text = re.sub(r"\n+\s*\d+\s*/\s*\d+\s*(?=\n|$)", "\n", text)
    text = re.sub(r"(?m)^\s*(Page|Стр\.?|стр\.?)\s*\d+\s*$", " ", text)
    text = re.sub(r"([^\W\d_])-\s*\n\s*([^\W\d_])", r"\1\2", text, flags=re.U)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # --- Обработка формул ---
    if replace_formulas_with_token:
        text = re.sub(r"\$.*?\$", f" {formula_token} ", text, flags=re.S)
        text = re.sub(r"\\\[.*?\\\]", f" {formula_token} ", text, flags=re.S)
        text = re.sub(r"\\\(.*?\\\)", f" {formula_token} ", text, flags=re.S)
        text = re.sub(
            r"\([^\n]{0,200}?[=+\-*/^_<>≡≤≥≈≠∑∏∫∂πσλθ±]\s*[^\n]{0,200}?\)",
            f" {formula_token} ",
            text,
        )
        if aggressive_math_removal:
            text = re.sub(
                r"[A-Za-z0-9_]*[=+\-*/^_<>∑∏∫∂πσλθ±≤≥≈≠][A-Za-z0-9_,\.\s\{\}\[\]\(\)]{0,200}",
                f" {formula_token} ",
                text,
            )

    if remove_math_symbols:
        math_symbols = r"[≡∀∃⊆⊂∪∩Δ∑∏⊕∞≤≥≈≠∂∇πσλθ±∫]"
        text = re.sub(math_symbols, " ", text)

    text = re.sub(r"[{}\[\]|]", " ", text)
    text = re.sub(r"\b(for|while|yield|return|do|end)\b", " ", text, flags=re.I)
    if remove_latin:
        text = re.sub(r"[A-Za-z]", " ", text)
    if remove_digits:
        text = re.sub(r"\d+", " ", text)

    # --- Удаление нумерации заголовков ---
    text = re.sub(r"(?m)^\s*(\d+(?:\.\d+)*\.)\s*", "", text)

    # --- Исправление разорванных строк ---
    lines = text.split("\n")
    merged_lines = []
    buffer = ""
    for line in lines:
        line = line.strip()
        if not line:
            if buffer:
                merged_lines.append(buffer)
                buffer = ""
            continue
        # Сливаем строки, если следующая начинается с маленькой буквы
        if buffer:
            if line[0].islower():
                buffer += " " + line
            else:
                merged_lines.append(buffer)
                buffer = line
        else:
            buffer = line
    if buffer:
        merged_lines.append(buffer)

    text = "\n".join(merged_lines)

    # --- Форматирование FORMULA ---
    text = re.sub(r"(?<!\[)FORMULA(?!\])", "[FORMULA]", text, flags=re.IGNORECASE)

    # --- Нормализация пробелов ---
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\n{2,}", "\n\n", text)

    return text.strip()

# -----------------------------
# 4. группировка страниц в контейнеры
# -----------------------------
def group_pages_to_containers(pdf_path: str) -> List[Dict[str, Any]]:
    containers = []
    current = None
    page_no = 0
    for page_layout in extract_pages(pdf_path):
        page_no += 1
        ptext = page_to_text(page_layout)
        ptext = re.sub(r'\n+\d+\s*/\s*\d+\s*$', '', ptext)
        first_lines = "\n".join([ln for ln in ptext.splitlines() if ln.strip()][:3])
        m = HEADER_RE.match(first_lines) if first_lines else None
        if m:
            sec = m.group(1).strip() if m.group(1) else f"no_{page_no}"
            title_text = m.group(2).strip() if m.group(2) else ""
            title_text = re.sub(r"\(\d+/\d+\)", "", title_text)  # убираем (1/2)
            brackets = m.group(3).strip() if m.group(3) else ""
            if current is not None:
                if current["section_id"] == sec:
                    current["page_texts"].append(ptext)
                else:
                    containers.append(current)
                    current = {
                        "section_id": sec,
                        "page_texts": [ptext],
                        "metadata": {
                            "section": title_text,
                            "subsection": "",
                            "brackets": brackets
                        }
                    }
            else:
                current = {
                    "section_id": sec,
                    "page_texts": [ptext],
                    "metadata": {"section": title_text, "subsection": "", "brackets": brackets}
                }
        else:
            if current is None:
                current = {
                    "section_id": "preface",
                    "page_texts": [ptext],
                    "metadata": {"section": "", "subsection": "", "brackets": ""}
                }
            else:
                current["page_texts"].append(ptext)

    if current is not None:
        containers.append(current)
    return containers

# -----------------------------
# 5. удаление дублирующихся заголовков
# -----------------------------
import re

import re

def remove_duplicate_headers(text):
    """
    Удаляет повторяющиеся заголовки в тексте.
    Заголовки определяются как строки, оканчивающиеся на цифры или двоеточие.
    """
    seen = set()
    result = []

    # Разделяем текст на строки
    lines = text.splitlines()
    for line in lines:
        # Убираем лишние пробелы
        clean_line = line.strip()
        # Считаем заголовком, если он короткий и начинается с цифры или содержит ключевые слова
        if re.match(r'^(\d+(\.\d+)*\.?\s*)', clean_line):
            if clean_line not in seen:
                seen.add(clean_line)
                result.append(clean_line)
        else:
            result.append(clean_line)

    return "\n".join(result)

def clean_text(text):
    """
    Дополнительно удаляет подряд идущие пустые строки
    """
    text = remove_duplicate_headings(text)
    # Убираем пустые строки подряд
    text = re.sub(r'\n\s*\n', '\n\n', text)
    return text


# 6. валидация текста
# -----------------------------
def validate_text(text: str):
    if len(text) < 1000:
        print("⚠️ Слишком короткий текст")
    if re.search(r"[^\x00-\x7Fа-яА-ЯёЁ0-9\s.,;:()\-]", text):
        print("⚠️ Подозрительные символы")
    if "  " in text:
        print("⚠️ Много двойных пробелов")

# -----------------------------
# 7. основной конвертер PDF в текст
# -----------------------------
def pdf_to_plain_text(pdf_path: str) -> str:
    containers = group_pages_to_containers(pdf_path)
    try:
        ocr_text = pdf_to_text_ocr(pdf_path)
    except Exception as e:
        print(f"[WARN] OCR failed: {e}")
        ocr_text = ""

    output_lines = []

    for cont in containers:
        full_text = "\n".join(cont["page_texts"])
        if full_text.count("FORMULA") > 3 and ocr_text:
            print("[INFO] Using OCR for this block")
            full_text = ocr_text

        clean_text = cleaning_text(full_text, replace_symbols_with_words=True)
        section_id = cont["section_id"]
        metadata = cont["metadata"]

        if section_id == "preface" and not metadata["section"]:
            output_lines.append(clean_text)
        else:
            title = f"{section_id}. {metadata['section']}"
            if metadata.get("brackets"):
                title += f" ({metadata['brackets']})"
            output_lines.append(title)
            output_lines.append(clean_text)

        output_lines.append("")

    full_text = "\n".join(output_lines).strip()
    full_text = remove_duplicate_headers(full_text)
    validate_text(full_text)
    return full_text

# -----------------------------
# 8. запуск
# -----------------------------
if __name__ == "__main__":
    pdf_file = "DM2024_module9.pdf"  # путь к PDF
    txt_file = "DM2024_module9.txt"

    plain_text = pdf_to_plain_text(pdf_file)
    with open(txt_file, "w", encoding="utf-8") as f:
        f.write(plain_text)

    print(f"Обработка завершена. Результат сохранён в {txt_file}")