import pickle

import re
from datetime import datetime
from typing import List, Dict

from langchain.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document
from langchain.vectorstores import Chroma

CHAT_FILE = "/home/ubuntu/divine-whatsapp/_chat.txt"
PERSISTENT_DIRECTORY = "data/whatsapp_chroma"
PATTERN = r'\[(\d+/\d+/\d+), (\d+:\d+:\d+)\] ([^:]+): (.*)'


def ingest_docs():
    """Get documents from chat history."""
    f = open(CHAT_FILE, "r")
    s = f.read()
    f.close()
    l = s.split('\n')
    documents = whatsapp_to_documents(l)
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=PERSISTENT_DIRECTORY
    )
    vectorstore.persist()

    # Save vectorstore
    with open("vectorstore_whatsapp.pkl", "wb") as f:
        pickle.dump(vectorstore, f)


def whatsapp_to_documents(chat: List[str]) -> List[Document]:
    i = 1
    msg = chat[0]
    documents = []
    while i < len(chat):
        matches = re.search(PATTERN, chat[i])
        if matches:
            parsed_line = parse_line(msg)
            documents.append(
                Document(
                    page_content=parsed_line['message'],
                    metadata={"source": CHAT_FILE, "sender": parsed_line['sender']}
                )
            )
            msg = chat[i]
        else:
            msg = msg + chat[i]
        i = i + 1

    return documents


def parse_line(whatsapp_chat_line: str) -> Dict:
    matches = re.search(PATTERN, whatsapp_chat_line)
    if matches:
        date_str = matches.group(1)
        time_str = matches.group(2)
        sender = matches.group(3)
        message = matches.group(4)
        datetime_obj = datetime.strptime(f"{date_str} {time_str}", "%m/%d/%y %H:%M:%S")
        epoch_milliseconds = int(datetime_obj.timestamp())
        return {
            'id': epoch_milliseconds,
            'sender': sender,
            'message': message
        }
    else:
        raise Exception("Unable to parse whatapp chat line.", whatsapp_chat_line)


if __name__ == "__main__":
    ingest_docs()
