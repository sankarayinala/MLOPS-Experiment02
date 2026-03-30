pipeline {
    agent any

    stages {
        stage('Checkout') {
            steps {
                echo 'Checking out code...'
                checkout scmGit(
                    branches: [[name: '*/main']], 
                    userRemoteConfigs: [[credentialsId: 'github-token', url: 'https://github.com/sankarayinala/MLOPS-Experiment02.git']]
                )
            }
        }

        stage('Test Docker Access') {
            steps {
                sh 'docker --version'
                sh 'whoami && id'
                sh 'docker info | head -n 20'
            }
        }

        stage('Build Docker Image') {
            steps {
                echo 'Building custom Jenkins image...'
                sh 'docker build -t jenkins-anime:latest .'
            }
        }
    }

    post {
        success { echo '✅ Pipeline Success!' }
        failure { echo '❌ Pipeline Failed!' }
    }
}