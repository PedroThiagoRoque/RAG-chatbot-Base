import os
import chromadb
import openai
import PyPDF2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
import threading
from textwrap import fill

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()

# Configurar chave da API OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

# Inicializar o cliente do banco de dados vetorial Chroma como persistente
chroma_client = None
persistent_path = None

# Função para extrair texto dos PDFs
#def extract_text_from_pdf(file_path):
#    text = ""
#    with open(file_path, 'rb') as file:
#        reader = PyPDF2.PdfReader(file)
#        for page_num in range(len(reader.pages)):
#            text += reader.pages[page_num].extract_text()
#    return text

# Função para extrair texto dos PDFs, ignorando caracteres inválidos
def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page_num in range(len(reader.pages)):
            try:
                # Extração de texto com fallback para evitar erro de codificação
                page_text = reader.pages[page_num].extract_text()
                if page_text:
                    # Remove ou substitui caracteres não UTF-8
                    page_text = page_text.encode('utf-8', errors='ignore').decode('utf-8')
                    text += page_text
            except Exception as e:
                print(f"Erro ao processar a página {page_num} do arquivo {file_path}: {e}")
    return text

# Função para dividir o texto em chunks (trechos menores, no máximo 500 caracteres)
def chunk_text(text, max_length=500):
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_length):
        chunk = " ".join(words[i:i + max_length])
        chunks.append(chunk)
    return chunks

# Preparar os dados: extrair texto dos PDFs, dividir em chunks e gerar embeddings
def process_pdfs(directory, status_label, progress_bar):
    global chroma_client
    try:
        status_label.config(text="Processando PDFs...")
        collection = chroma_client.get_or_create_collection(name="document_chunks")
        files = [f for f in os.listdir(directory) if f.endswith('.pdf')]
        progress_bar["maximum"] = len(files)
        progress_bar["value"] = 0

        for pdf_file in files:
            pdf_path = os.path.join(directory, pdf_file)
            status_label.config(text=f"Processando {pdf_file}...")

            # Extrair texto do PDF
            text = extract_text_from_pdf(pdf_path)
            chunks = chunk_text(text)

            # Gerar embeddings e adicionar ao banco de dados
            for chunk in chunks:
                embedding = openai.embeddings.create(model="text-embedding-ada-002", input=chunk)
                embedding_vector = embedding.data[0].embedding
                collection.add(ids=[str(hash(chunk + str(pdf_file)))], embeddings=[embedding_vector], documents=[chunk])


            progress_bar.step(1)
        status_label.config(text="Processamento concluído com sucesso!")
    except Exception as e:
        status_label.config(text=f"Erro ao processar PDFs: {e}")

# Função para adicionar um único PDF ao banco de dados
def add_single_pdf_to_database(pdf_path, status_label):
    global chroma_client
    try:
        if not chroma_client:
            raise Exception("Banco de dados persistente não está configurado.")

        status_label.config(text=f"Processando {os.path.basename(pdf_path)}...")
        collection = chroma_client.get_or_create_collection(name="document_chunks")

        # Extrair texto do PDF
        text = extract_text_from_pdf(pdf_path)
        chunks = chunk_text(text)

        # Gerar embeddings e adicionar ao banco de dados
        for chunk in chunks:
            embedding = openai.embeddings.create(model="text-embedding-ada-002", input=chunk)
            embedding_vector = embedding.data[0].embedding
            collection.add(ids=[str(hash(chunk + str(pdf_path)))], embeddings=[embedding_vector], documents=[chunk])

        status_label.config(text=f"{os.path.basename(pdf_path)} adicionado ao banco com sucesso!")
    except Exception as e:
        status_label.config(text=f"Erro ao adicionar PDF: {e}")


# Função para recuperar chunks relevantes com base em uma consulta
def retrieve_relevant_chunks(query, top_k=5):
    query_embedding = openai.embeddings.create(model="text-embedding-ada-002", input=query)
    query_vector = query_embedding.data[0].embedding
    collection = chroma_client.get_collection(name="document_chunks")
    results = collection.query(query_embeddings=[query_vector], n_results=top_k)
    return results["documents"]

# Função para gerar uma resposta com base nos chunks relevantes
def generate_response(query, status_label):
    try:
        status_label.config(text="Buscando resposta...")
        relevant_chunks = retrieve_relevant_chunks(query)
        #print(relevant_chunks) 
        context = "\n".join(relevant_chunks[0])

        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
            {"role": "system", "content": (
                "Você é um assistente especializado em eficiência energética e temas relacionados. Seu objetivo é criar respostas explicativas ao usuário utilizando as informações presentes nos chunks fornecidos no contexto. Você pode reformular o conteúdo dos chunks, porém sem se afastar de seu significado original e sem incluir informações não suportadas pelo contexto. Se não for possível responder com base nos chunks, responda: 'Esta informação não consta nos artigos armazenados'. Sempre que o conteúdo ou a pergunta estiverem em outro idioma, traduza para o Português do Brasil antes de responder."
            )},
                {"role": "user", "content": f"Contexto: {context}\n\nPergunta: {query}"}
            ],
            temperature=0.5,
            max_tokens=150
        )
        status_label.config(text="Resposta gerada com sucesso!")
        return response.choices[0].message.content
    except Exception as e:
        status_label.config(text=f"Erro ao gerar resposta: {e}")
        return ""

# Configurar UI principal
def main():
    global chroma_client, persistent_path
    root = tk.Tk()
    root.title("Chatbot com RAG")
    root.geometry("600x600")

    messages_frame = tk.Frame(root)
    messages_frame.pack(fill=tk.BOTH, expand=True)

    scrollbar = tk.Scrollbar(messages_frame)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    messages_list = tk.Text(messages_frame, yscrollcommand=scrollbar.set, wrap=tk.WORD, bg="black", fg="white", insertbackground="white")
    messages_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar.config(command=messages_list.yview)

    entry_frame = tk.Frame(root)
    entry_frame.pack(fill=tk.X)

    entry = tk.Entry(entry_frame)
    entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=5)

    def send_message():
        user_query = entry.get()
        if not user_query:
            return

        messages_list.insert(tk.END, f"Você: {fill(user_query, 80)}\n")
        entry.delete(0, tk.END)

        def fetch_response():
            response = generate_response(user_query, status_label)
            messages_list.insert(tk.END, f"Chatbot: {fill(response, 80)}\n")

        threading.Thread(target=fetch_response).start()

    send_button = tk.Button(entry_frame, text="Enviar", command=send_message)
    send_button.pack(side=tk.RIGHT, padx=5, pady=5)

    status_label = tk.Label(root, text="Pronto para uso", anchor="w")
    status_label.pack(fill=tk.X, pady=5)

    progress_bar = ttk.Progressbar(root, orient="horizontal", mode="determinate")
    progress_bar.pack(fill=tk.X, pady=5)

    def choose_pdf_directory():
        global pdf_directory
        pdf_directory = filedialog.askdirectory()
        if pdf_directory:
            threading.Thread(target=process_pdfs, args=(pdf_directory, status_label, progress_bar)).start()

    def choose_database_path():
        global chroma_client, persistent_path
        persistent_path = filedialog.askdirectory()
        if persistent_path:
            chroma_client = chromadb.PersistentClient(path=persistent_path)
            status_label.config(text=f"Banco de dados carregado de: {persistent_path}")

    def add_single_pdf():
        pdf_path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
        if pdf_path:
            threading.Thread(target=add_single_pdf_to_database, args=(pdf_path, status_label)).start()

    folder_button = tk.Button(root, text="Selecionar Pasta de PDFs", command=choose_pdf_directory)
    folder_button.pack(pady=5)

    db_button = tk.Button(root, text="Selecionar BD Persistente", command=choose_database_path)
    db_button.pack(pady=5)

    add_pdf_button = tk.Button(root, text="Adicionar PDF", command=add_single_pdf)
    add_pdf_button.pack(pady=5)

    root.mainloop()

if __name__ == "__main__":
    main()
