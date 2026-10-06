import io
import re
import csv
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from rapidocr import RapidOCR
from openpyxl import load_workbook
from openpyxl.utils.cell import coordinate_from_string, column_index_from_string, get_column_letter

st.set_page_config(page_title='測定値・文字列OCR', page_icon='🔢', layout='wide')
st.title('🔢 「XX.X」を文字列のまま読み取ってExcelへ')
st.caption('桁別モデルは使いません。修正した結果は確認済みデータとしてZIP保存できます。')

@st.cache_resource
def get_ocr():
    return RapidOCR()

OCR = get_ocr()
if 'corrections' not in st.session_state:
    st.session_state.corrections = {}


def timekey(name):
    m = re.search(r'_(\d+(?:\.\d+)?)s(?:\.[^.]+)?$', name, re.I)
    return (0, float(m.group(1))) if m else (1, name.lower())


def valid_value(value):
    value = str(value).strip()
    return bool(re.fullmatch(r'\d{2}\.\d', value)) and 20.0 <= float(value) <= 99.9


def clean_text(text):
    table = str.maketrans({'O':'0','o':'0','I':'1','l':'1','|':'1','S':'5','s':'5','B':'8','，':'.','。':'.',',':'.'})
    return str(text).translate(table)


def candidates_from_text(text, score, source):
    text_clean = clean_text(text)
    found = []
    for match in re.finditer(r'(\d{2})\s*\.\s*(\d)', text_clean):
        value = float(match.group(1) + '.' + match.group(2))
        if 20.0 <= value <= 99.9:
            found.append((value, float(score) + 0.20, source, text))
    digits = ''.join(re.findall(r'\d', text_clean))
    if len(digits) >= 3:
        for index in range(len(digits) - 2):
            value = float(digits[index:index+2] + '.' + digits[index+2])
            if 20.0 <= value <= 99.9:
                found.append((value, float(score) + (0.05 if index == 0 else 0), source, text))
    return found


def read_variant(name, image, weight):
    enlarged = cv2.resize(image, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    result = OCR(enlarged)
    rows = []
    if result is not None and result.txts is not None:
        rows = [(str(text), float(score)) for text, score in zip(result.txts, result.scores)]
    candidates = []
    for text, score in rows:
        candidates.extend(candidates_from_text(text, min(1.0, score * weight), name))
    return rows, candidates


def recognize(rgb):
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    height, width = bgr.shape[:2]
    center = bgr[int(height*0.38):int(height*0.68), int(width*0.20):int(width*0.78)]
    lab = cv2.cvtColor(center, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    enhanced_l = cv2.createCLAHE(2.0, (8, 8)).apply(l_channel)
    enhanced = cv2.cvtColor(cv2.merge([enhanced_l, a_channel, b_channel]), cv2.COLOR_LAB2BGR)
    variants = [('画像全体', bgr, 1.00), ('中央領域', center, 1.08), ('中央・補正', enhanced, 1.04)]
    raw_results = []
    all_candidates = []
    for name, image, weight in variants:
        rows, candidates = read_variant(name, image, weight)
        raw_results.append((name, rows))
        all_candidates.extend(candidates)
    if not all_candidates:
        return '', 0.0, raw_results, center, []
    grouped = {}
    for value, score, source, text in all_candidates:
        grouped.setdefault(value, []).append((score, source, text))
    ranked = []
    for value, evidence in grouped.items():
        scores = sorted([item[0] for item in evidence], reverse=True)
        combined = scores[0] + 0.12 * max(0, len(evidence)-1) + 0.05 * sum(scores[1:3])
        ranked.append((combined, value, evidence))
    ranked.sort(reverse=True)
    best = ranked[0]
    return f'{best[1]:.1f}', min(1.0, best[0]), raw_results, center, ranked[:5]


def correction_zip():
    output = io.BytesIO()
    rows = []
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as archive:
        for filename, record in st.session_state.corrections.items():
            rows.append({
                'filename': filename,
                'ocr_result': record['ocr_result'],
                'correct_value': record['correct_value'],
                'confidence': record['confidence'],
            })
            archive.writestr(f'images/{filename}', record['image_bytes'])
        csv_buffer = io.StringIO()
        writer = csv.DictWriter(csv_buffer, fieldnames=['filename','ocr_result','correct_value','confidence'])
        writer.writeheader()
        writer.writerows(rows)
        archive.writestr('corrections.csv', csv_buffer.getvalue().encode('utf-8-sig'))
    return output.getvalue()


def cellok(value):
    return bool(re.fullmatch(r'[A-Za-z]{1,3}[1-9][0-9]*', value.strip()))

excel = st.file_uploader('1. 入力先Excel', type=['xlsx'])
files = sorted(st.file_uploader('2. 加工前写真を選択', type=['png','jpg','jpeg','webp'], accept_multiple_files=True) or [], key=lambda file: timekey(file.name))
if files:
    st.info('時間順：' + ' → '.join(file.name for file in files))

rows = []
for index, file in enumerate(files):
    image_bytes = file.getvalue()
    try:
        rgb = np.array(Image.open(io.BytesIO(image_bytes)).convert('RGB'))
    except Exception as error:
        st.error(f'{file.name}を開けません：{error}')
        continue
    predicted, confidence, raw, center, ranked = recognize(rgb)
    left, right = st.columns([1, 2])
    left.image(rgb, caption=file.name, width='stretch')
    corrected = right.text_input('認識結果（間違っていれば修正）', predicted, key=f'value_{index}_{file.name}')
    right.write(f'OCR候補の確度：{confidence:.2f}')
    if not predicted:
        right.error('20.0～99.9の候補を取得できませんでした。')
    elif confidence < 0.70:
        right.warning('候補の一致が弱いため確認してください。')
    if right.button('修正データとして保存', key=f'save_{index}_{file.name}'):
        if not valid_value(corrected):
            right.error('20.0～99.9のXX.X形式で入力してください。')
        elif corrected == predicted:
            right.info('OCR結果と同じため、修正データには追加しません。')
        else:
            st.session_state.corrections[file.name] = {
                'ocr_result': predicted,
                'correct_value': corrected,
                'confidence': round(confidence, 4),
                'image_bytes': image_bytes,
            }
            right.success('元画像と修正値を保存しました。')
    with right.expander('OCRの読み取り内容'):
        right.image(cv2.cvtColor(center, cv2.COLOR_BGR2RGB), caption='中央の探索領域', width='stretch')
        for name, items in raw:
            right.write(name + '：' + ('、'.join(f'{text} ({score:.2f})' for text, score in items) if items else '文字なし'))
        if ranked:
            right.write('数値候補：' + ', '.join(f'{value:.1f}' for _, value, _ in ranked))
    rows.append({'filename': file.name, 'value': corrected, 'confidence': round(confidence, 3)})

if st.session_state.corrections:
    st.subheader('保存した修正データ')
    summary = [{k:v for k,v in record.items() if k != 'image_bytes'} | {'filename': filename} for filename, record in st.session_state.corrections.items()]
    st.dataframe(pd.DataFrame(summary), width='stretch')
    st.download_button('修正データ一式をダウンロード', correction_zip(), 'ocr_corrections.zip', 'application/zip')
    if st.button('保存した修正データをクリア'):
        st.session_state.corrections = {}
        st.rerun()

if rows:
    st.dataframe(pd.DataFrame(rows), width='stretch')
    if excel:
        workbook = load_workbook(io.BytesIO(excel.getvalue()))
        sheet_name = st.selectbox('入力シート', workbook.sheetnames)
        start = st.text_input('開始セル', 'C4')
        if cellok(start):
            letters, first_row = coordinate_from_string(start.upper())
            column = column_index_from_string(letters)
            worksheet = workbook[sheet_name]
            invalid = []
            for offset, row in enumerate(rows):
                try:
                    value = float(row['value'])
                    if not 20.0 <= value <= 99.9:
                        raise ValueError
                    cell = worksheet.cell(first_row + offset, column, value)
                    cell.number_format = '0.0'
                except Exception:
                    invalid.append(row['filename'])
            if invalid:
                st.error('有効な数値にできない画像：' + '、'.join(invalid))
            else:
                output = io.BytesIO()
                workbook.save(output)
                end = f'{get_column_letter(column)}{first_row + len(rows) - 1}'
                st.success(f'{start.upper()}:{end}へ入力しました')
                st.download_button('入力済みExcelをダウンロード', output.getvalue(), f'{Path(excel.name).stem}_入力済み.xlsx')
