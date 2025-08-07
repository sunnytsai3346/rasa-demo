#### rasa-demo (Rasa-powered RAG chatbot with LLM integration) ####
----------------------------------------------------

Description:
A demo chatbot combining Rasa NLU, LangChain, Qdrant, and LLaMA3 (local via Ollama) to provide conversational AI using private documents.

README Sections:

🔍 Overview — chatbot with retrieval-augmented generation; integrated with PDF/document semantic search.

🛠 Technology Stack — Rasa, Python, LangChain, Ollama, FAISS/Qdrant.

🚧 Features — multilingual NLU, semantic retrieval, custom actions.

➡️ Architecture Diagram — show pipeline: user input → Rasa → vector search → LLM response.

🪄 Setup Instructions — cloning, Docker build, model training, run: rasa train & docker-compose up.

🎬 Demo Usage — example queries and responses.

🧪 Tests & Metrics — todo.

📈 Future Directions — fine-tuning LLaMA, multi-agent workflows, agentic memory.

----------------------------------------------------
- 1️⃣ Install Rasa (pip install rasa)
- 2️⃣ Initialize project (rasa init)
- 3️⃣ Add training data (data/nlu.yml)
- 4️⃣ Define responses (domain.yml)
- 5️⃣ Train the model (rasa train) ***    ( with rasa data validate to check train data, if error , need to fix If there are errors, fix the training data (nlu.yml, stories.yml).)
- 6️⃣ Test with (rasa shell)
- 7️⃣ Add stories (data/stories.yml)
- 8️⃣ Use actions (actions.py)
- 9. Enabled API Mode (rasa run --enable-api --cors "*")
-10. rasa run actions  ( start this actions in another terminal Start the Actions Server)
-11  rasa stop  ( if need to stop rasa)

trouble shooting:
##
installed Rasa inside a virtual environment, but forgot to activate it. will show error " rasa : The term 'rasa' is not recognized "
If Rasa was installed using pip inside a virtual environment (like venv or conda), you need to activate that environment before running rasa commands.
##
pip install --upgrade rasa rasa-sdk sanic
pip install --upgrade tensorflow
python -c "import tensorflow as tf; print(tf.__version__)"  -- to check if it's installed correctly:

1. pip uninstall numpy tensorflow tensorflow-intel scipy rasa
2. pip install numpy==1.23.5 tensorflow==2.12.0 tensorflow-intel==2.12.0 scipy==1.10.1
3. pip install rasa==3.6.21
4. check version : pip list | grep -E "rasa|numpy|tensorflow|scipy"  ( in git bash) , if in windows use pip list |findstr "rasa numpy tensorflow"

Check Rasa Dependencies

The rapidfuzz dependency might be required by a specific version of a package Rasa uses (e.g., cleo or poetry). Ensure all Rasa dependencies are up-to-date:

bash

Copy
pip install --upgrade rasa rasa-sdk
pip install pymupdf
pip install --upgrade rasa rasa-sdk
pip uninstall spacy thinc
pip install spacy==3.5.4
python -m spacy download en_core_web_md
pip install langdetect



rm -rf models/*

rasa --version
Rasa Version      :         3.6.21
Minimum Compatible Version: 3.6.21
Rasa SDK Version  :         3.6.2
Python Version    :         3.10.11ras

ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.                                                    
rasa 3.6.21 requires numpy<1.25.0,>=1.19.2; python_version >= "3.8" and python_version < "3.11", but you have numpy 2.2.5 which is incompatible.
rasa 3.6.21 requires pydantic<1.10.10, but you have pydantic 2.11.4 which is incompatible.
scipy 1.10.1 requires numpy<1.27.0,>=1.19.5, but you have numpy 2.2.5 which is incompatible.
tensorflow-intel 2.12.0 requires numpy<1.24,>=1.22, but you have numpy 2.2.5 which is incompatible.
