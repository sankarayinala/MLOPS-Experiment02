📘 Anime Recommendation System — Full Documentation
✅ Overview
This project is a hybrid AI-powered recommender system combining collaborative filtering, content-based filtering, and MMR diversification. It includes:
✅ FastAPI backend
✅ Streamlit UI
✅ Redis caching
✅ FAISS similarity engine
✅ Jenkins CI/CD
✅ Docker deployment

✅ Architecture Diagram
          ┌───────────┐
          │ Streamlit │
          └─────┬─────┘
                │ REST
                ▼
          ┌───────────┐
          │  FastAPI  │
          ├───────────┤
          │ Hybrid ML │
          │  Engine   │
          └─────┬─────┘
                │
         ┌──────┴──────┐
         │  Redis Cache│
         └──────┬──────┘
                │
        ┌───────┴────────┐
        │ FAISS Indexes  │
        │ Embeddings     │
        └────────────────┘


✅ How to Install Docker (Cited)
Docker Desktop installation steps for Windows:
✅ System requirements, WSL2 support, download links, installation flow documented in Docker official docs 
✅ Step-by-step Windows guide from TheLinuxCode (2026 updated tutorial) [docs.docker.com] [thelinuxcode.com]

✅ How to Install Jenkins (Cited)
✅ Windows MSI installation steps: setup wizard, service credentials, admin password, plugin installation 
✅ Alternative WAR file method documented in GeeksForGeeks guide (2026) [jenkins.io] [geeksforgeeks.org]

✅ Running with Docker
docker-compose build
docker-compose up -d

Backend → http://localhost:8000
Streamlit UI → http://localhost:8501

✅ Running CI/CD with Jenkins

Install Jenkins
Add Docker plugin
Add Git credentials
Create Pipeline → point to Jenkinsfile


✅ 6. Final Deliverables Summary
This response includes:
✅ Full project documentation
✅ Parameterized configuration template
✅ Dockerfiles for backend + UI
✅ docker-compose.yml
✅ Jenkinsfile CI/CD pipeline
✅ README.md with installation steps
✅ Citations for Docker & Jenkins installation requirements
