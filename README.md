# 1. function

a LLM test project for some RAG and SQL Agent test purpose. 
Have fun, enjoy life !

# 2. intro

 (1) http_rag.py, a RAG demo, you can input question in URI `http://localhost:19000`;

 (2) sql_agent.py is a SQL agent demo.

# 3. build docker file

```sh
docker build -f ./myDockerfile ./ -t llm_agent:1.0
```

# 4. run

package all your pip package in a docker images named llm_agent with version 1.0.

put all your python script file in dir /data/llm_agent, and run it as followling:

```sh
docker run -dit --name test --rm -p 19001:19000 -v /data/llm_agent:/opt/app llm_agent:1.0
# maybe you can try set entrypoint to boot procedure automatically
docker run -dit --name test --rm --entrypoint "/opt/app/start.sh" -p 19001:19000 -v /data/llm_agent:/opt/app llm_agent:1.0
```

