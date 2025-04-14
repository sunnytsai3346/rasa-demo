- 1️⃣ Install Rasa (pip install rasa)
- 2️⃣ Initialize project (rasa init)
- 3️⃣ Add training data (data/nlu.yml)
- 4️⃣ Define responses (domain.yml)
- 5️⃣ Train the model (rasa train) ***    ( with rasa data validate to check train data, if error , need to fix If there are errors, fix the training data (nlu.yml, stories.yml).)
- 6️⃣ Test with (rasa shell) *** 
- 7️⃣ Add stories (data/stories.yml)
- 8️⃣ Use actions (actions.py)
- 9. Enabled API Mode (rasa run --enable-api --cors "*")  rasa run -m models --enable-api --endpoints endpoints.yml --cors "*"
-10. rasa run actions  ( start this actions in another terminal Start the Actions Server) -- in another terminal , to hook customized actions
-11  rasa stop  ( if need to stop rasa)
-12  pip install fuzzywuzzy[speedup]

trouble shooting:
pip install --upgrade rasa rasa-sdk sanic
pip install --upgrade tensorflow
python -c "import tensorflow as tf; print(tf.__version__)"  -- to check if it's installed correctly:

1. pip uninstall numpy tensorflow tensorflow-intel scipy rasa
2. pip install numpy==1.23.5 tensorflow==2.12.0 tensorflow-intel==2.12.0 scipy==1.10.1
3. pip install rasa==3.6.21
4. check version : pip list | grep -E "rasa|numpy|tensorflow|scipy"  ( in git bash) , if in windows use pip list |findstr "rasa numpy tensorflow"
5. check fuzzywuzzy : pip list | findstr fuzzy





rm -rf models/*

rasa --version
Rasa Version      :         3.6.21
Minimum Compatible Version: 3.6.21
Rasa SDK Version  :         3.6.2
Python Version    :         3.10.11ras
