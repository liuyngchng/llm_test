#1. function
a LLM test project for som RAG and SQL Agent test purpose. 
Have fun, enjoy life !

#2. intro

 (1)http_rag.py, a RAG demo, you can input question in URI `http://localhost:19000`;
 
 (2)sql_agent.py is a SQL agent demo.

#3. build docker file

```sh
docker build -f ./myDockerfile ./ -t myimg:1.0.0
```

#4. docker run

package all your pip package in a docker images name llm with version 1.0.

put all your python script file in dir /data/llm_agent, and run it like followling.

```sh
docker run -dit --name test -p 19001:19000 -v /data/llm_agent:/opt/app llm:1.0
```
