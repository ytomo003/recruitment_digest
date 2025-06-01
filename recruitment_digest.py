import os
import openai
import time
import traceback
import json
import sys
import datetime
import re
import requests
import gspread
import boto3
import logging
import textwrap
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv
from oauth2client.service_account import ServiceAccountCredentials
from googleapiclient.discovery import build
from botocore.exceptions import NoCredentialsError
from langchain.document_loaders import TextLoader
from langchain.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough


# ロギングの設定
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# 環境変数の読み込み
load_dotenv('.env')

# 設定値の定義
API_KEY = os.getenv('OPEN_API_KEY')
S3_CONFIG = {
    'aws_access_key_id': os.getenv('AWS_ACCESS_KEY_ID'),
    'aws_secret_access_key': os.getenv('AWS_SECRET_ACCESS_KEY'),
    'region_name': os.getenv('AWS_REGION_NAME')
}
GOOGLE_CREDENTIALS_PATH = 'sample.json'
SPREADSHEET_ID = 'XXXXX'

# APIクライアントの初期化
openai.api_key = API_KEY
s3 = boto3.client('s3', **S3_CONFIG)

"""
Google APIクライアントの初期化を行う関数

Returns:
    tuple: シートクライアントとドキュメントサービスのタプル
"""
scope = ["https://www.googleapis.com/auth/spreadsheets.readonly",
         "https://www.googleapis.com/auth/documents.readonly"]
credentials = ServiceAccountCredentials.from_json_keyfile_name(
    GOOGLE_CREDENTIALS_PATH, scope)
sheet_client = gspread.authorize(credentials)
docs_service = build('docs', 'v1', credentials=credentials)


def request_datasource(url):
    """
    指定されたURLからデータを取得する関数

    Args:
        url (str): データソースのURL

    Returns:
        dict: 取得したJSONデータ、エラー時はNone
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"データソースの取得でエラーが発生しました: {e}")
        print(traceback.format_exc())
        return None


def request_document_to_jsonurl(url):
    """
    Google DocsのURLからドキュメントの内容を取得する関数

    Args:
        url (str): Google DocsのURL
        docs_service: Google Docs APIサービス

    Returns:
        str: ドキュメントの内容、エラー時はNone
    """
    try:
        document_id = re.search(r'/document/d/([a-zA-Z0-9-_]+)', url)
        if not document_id:
            raise ValueError("ドキュメントIDがURLから見つかりませんでした")

        document_id = document_id.group(1)
        doc = docs_service.documents().get(documentId=document_id).execute()
        doc_content = doc.get('body', {}).get('content', [])

        text_content = ""
        for element in doc_content:
            if 'paragraph' in element:
                for text_run in element['paragraph']['elements']:
                    if 'textRun' in text_run:
                        text_content += text_run['textRun']['content']

        return text_content
    except Exception as e:
        print(f"ドキュメント内容の取得でエラーが発生しました: {e}")
        print(traceback.format_exc())
        return None


def request_s3_data(url):
    """
    S3のURLからプリサインドURLを生成する関数

    Args:
        url (str): S3のURL

    Returns:
        str: 生成されたプリサインドURL、エラー時はNone
    """
    match = re.match(r'https://([^.]+)\.s3\.[^.]+\.amazonaws\.com/(.+)', url)
    if not match:
        print("無効なS3のURL形式です")
        return None

    bucket_name, object_key = match.groups()
    try:
        # S3からオブジェクトを取得
        response = s3.get_object(Bucket=bucket_name, Key=object_key)
        # JSONデータを取得
        json_data = json.loads(response['Body'].read().decode('utf-8'))
        return json_data
    except NoCredentialsError:
        print("AWS認証情報が見つかりません")
        return None


def format_large_text(text, max_length=2048):
    # テキストを指定した最大長に分割
    chunks = textwrap.wrap(text, width=max_length, break_long_words=False)

    # 各チャンクを整形
    formatted_chunks = [format_text_chunk(chunk) for chunk in chunks]

    # 整形されたチャンクを結合
    formatted_text = "\n".join(formatted_chunks)
    return formatted_text


def format_text_chunk(chunk):
    prompt = f"この文に句読点や改行を追加し、話題の変化を基準にパラグラフに分けて敬体で出力してください。元の文章の文字数を保存すること。{chunk}"

    # パラメータの定義
    parameters = {
        'engine': 'gpt-4o-mini',
        'max_tokens': 2048,
        'stop': None,
        'temperature': 0.5,
    }

    response = openai.chat.completions.create(
        model=parameters['engine'],
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=parameters['max_tokens'],
        temperature=parameters['temperature'],
        stop=parameters['stop'],
    )

    return response.choices[0].message.content.strip()


def text_splits(file_path, chunk_size=1000, chunk_overlap=0):
    """
    テキストファイルを分割する関数

    Args:
        file_path (str): テキストファイルのパス
    """
    # テキストローダーの初期化
    loader = TextLoader(file_path)

    # ドキュメントの読みこみ
    documents = loader.load()

    # チャンクサイズの制限を下回るまで再帰的に分割するテキストスプリッターのインポート
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    # テキストスプリッターの初期化
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # テキストをチャンクに分割
    return text_splitter.split_documents(documents)


def init_vector_store(documents):
    """
    ベクトルストアを初期化する関数

    Args:
        documents (list): ドキュメントのリスト

    Returns:
        FAISS: 初期化されたベクトルストア
    """
    logging.info(f"ベクトルストアの初期化を開始: ドキュメント数={len(documents)}")

    embeddings = OpenAIEmbeddings(api_key=API_KEY)
    vector_store = FAISS.from_documents(documents, embeddings)

    logging.info("ベクトルストアの初期化が完了しました。")
    return vector_store


def generate_text(company_name, file_path, output_file_path):
    """
    テキストを生成し、ファイルに書き込む関数

    Args:
        company_name (str): 会社名
        file_path (str): 入力ファイルのパス（テキストファイル）
        output_file_path (str): 出力ファイルのパス（テキストファイル） 
    """
    logging.info(f"テキスト生成開始: 会社名={company_name}")
    documents = text_splits(file_path)

    logging.info(f"テキスト分割完了: 分割数={len(documents)}")

    # ベクトルストアの初期化
    vector_store = init_vector_store(documents)
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    output_parser = StrOutputParser()

    prompt = ChatPromptTemplate.from_template(

        """
        以下のcontextだけに基づいて次の命令を実行してください
        ### 参照情報
        {context}
        
        {question}
        
        # 作成した文章
        """
    )
    llm = ChatOpenAI(api_key=API_KEY, model_name="gpt-4o",
                     temperature=0, max_tokens=4096)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | output_parser
    )

    # 求人メディア記事を生成
    result = chain.invoke('''Please consider additional information to convey the attractiveness of this company. Enhance the job posting by fleshing out the company's overview, strengths, corporate culture, and future vision in a clear and appealing manner that resonates with job seekers.
        The target audience includes those looking for better job opportunities and people researching job postings.
        Please write in formal language.
        Answer should be in Japanese.

        # Instructions
        ## Basic Information
        Please provide the following basic information:
        - Job Title: [Job Title]
        - Employment Type: [Full-time, Contract, etc.]
        - Work Location: [Address]
        - Working Hours: [Working Hours]
        - Salary: [Salary Range]
        - Required Skills: [Skills]

        ## Company Overview
        Answer the following questions to supplement the company's strengths and vision.
        1. **Company Strengths**: What differentiates the company from others? Are there any notable achievements or projects to highlight?
        2. **Corporate Culture**: What is the workplace atmosphere like, and how do employees communicate with each other? How do diversity and work environment contribute to employee satisfaction?
        3. **Growth and Future Potential**: What is the company’s position in the market, and what vision does it have for future business expansion?

        ## Key Selling Points
        Based on the following questions, describe specific points that will attract job seekers.
        1. **Appeal of the Job**: Explain the job's fulfillment and the interesting aspects of the projects involved.
        2. **Career Path**: What career growth opportunities are available if one applies for this position?
        3. **Work-Life Balance**: What policies are in place regarding remote work, flex-time, and employee benefits?

        ## Message to Job Seekers
        1. Please provide additional information from the hiring manager about the passion they wish to convey to applicants and the type of candidate they are looking for.
        ''')
    print(result)
    # content = response.content

    write_text = generate_text_format(company_name, '', result)

    with open(output_file_path, "w", encoding="utf-8") as file:
        file.write(write_text)
    logging.info(f"ファイル作成完了: {output_file_path}")


def generate_text_format(url, header, text):
    """
    テキストの形式を整える関数

    Args:
        url (str): 会社のURL
        header (str): ヘッダーテキスト
        text (str): 本文テキスト

    Returns:
        str: 整形されたテキスト
    """
    return f'''会社ホームページ：{url}

{header}

本文ここから
{text}
'''


def init_data(hellowork_document_url, place_file_name, company_info_file_name):
    # ハローワークのデータソース
    if hellowork_document_url:  # hellowork_data_urlが空でない場合
        hellowork_data_url = request_document_to_jsonurl(
            hellowork_document_url)
        helloworks = request_s3_data(hellowork_data_url)

        if '仕事内容' not in helloworks:
            helloworks = None
    else:
        helloworks = None

    # GoogleMapsAPIのデータソース
    if place_file_name:  # place_file_nameが空でない場合
        places = request_s3_data(
            os.getenv('AWS_PLACE_URL') + place_file_name + '.json')
    else:
        places = None

    # 会社情報のデータソース
    if company_info_file_name:  # company_info_file_nameが空でない場合
        companies = request_s3_data(
            os.getenv('AWS_TARGET_URL') + company_info_file_name)
    else:
        companies = None

    return helloworks, places, companies


def main():
    """
    メイン実行関数
    S3からデータを取得し、テキスト生成を実行
    """
    try:
        # 最初のシートにアクセス
        sheet = sheet_client.open_by_key(SPREADSHEET_ID).sheet1
        # 会社名のデータソース
        company_names = sheet.col_values(5)

        # ===========デバッグ
        # testinndex = 0
        # paths = ['recruitment.txt']

        for i in range(len(company_names)):
            target_column = i+1

            # データがない場合はスキップ
            if company_names[target_column] == None:
                continue

            hellowork_column_values = sheet.col_values(3)
            place_column_values = sheet.col_values(12)
            company_info_column_values = sheet.col_values(13)

            # JSONデータをS3から取得
            helloworks, places, companies = init_data(
                hellowork_column_values[target_column], place_column_values[target_column], company_info_column_values[target_column])
            logging.info(
                f"メイン処理開始: 会社名={company_names[target_column]}")

            # today = datetime.date.today().strftime('%Y%m%d')
            now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

            # GPTで校正した後の文章
            output_file_path = f'{now}_{company_names[target_column]}GPT.txt'
            output_media_file_path = f"{now}_{company_names[target_column]}GPT_media.txt"

            formatted_hellowork_text = ''
            formatted_place_text = ''
            formatted_company_text = ''

            # テキストを整形
            if helloworks:
                formatted_hellowork_text = format_large_text(
                    str(helloworks), max_length=1024)
                formatted_hellowork_text = f'''
                #ハローワーク情報
                {formatted_hellowork_text}'''
                print(formatted_hellowork_text)
            else:
                logging.info(
                    f"ハローワーク情報がありません: 会社名={company_names[target_column]}")
                # 求人がない場合ファイルに書き込んでスキップ
                with open(output_media_file_path, "w", encoding="utf-8") as file:
                    file.write('ハローワーク情報がありません')
                logging.info(f"ファイル作成完了: {output_file_path}")

                continue

            # reviewのtextを取得
            if places:
                if 'reviews' in places['places'][0]:
                    reviews = places['places'][0]['reviews']
                    review_texts = [review.get('text', {}).get(
                        'text', '') for review in reviews]
                    formatted_place_text = format_large_text(
                        str(review_texts), max_length=1024)

                    formatted_place_text = f'''
                    #レビュー情報
                    {formatted_place_text}
                    '''
                    print(formatted_place_text)

            if companies:
                formatted_company_text = format_large_text(
                    str(companies), max_length=1024)
                formatted_company_text = f'''
                #会社ホームページ情報
                {formatted_company_text}
                '''
                print(formatted_company_text)

            # 整形されたテキストをファイルに書き込む
            with open(output_file_path, "w", encoding="utf-8") as file:
                file.write(formatted_hellowork_text +
                           formatted_place_text + formatted_company_text)

            # ===========デバッグ
            # path = paths[testinndex]
            # with open(path, 'r', encoding='utf-8') as f:
            #     content = f.read()
            # testinndex+1

            try:
                generate_text(company_names[target_column],
                              output_file_path, output_media_file_path)
            except Exception as e:
                logging.exception(f"GPTの実行でエラーが発生: {str(e)}")
                traceback.print_exc()

    except Exception as e:
        logging.exception(f"メイン実行でエラーが発生: {str(e)}")
        traceback.print_exc()


if __name__ == '__main__':
    main()
