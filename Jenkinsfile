pipeline {
    agent any

    options {
        skipDefaultCheckout true
    }

    stages {
        stage('Checkout') {
            steps {
                echo 'Checking out code from GitHub...'
                checkout scmGit(
                    branches: [[name: '*/main']], 
                    userRemoteConfigs: [[
                        credentialsId: 'github-token', 
                        url: 'https://github.com/sankarayinala/MLOPS-Experiment02.git'
                    ]]
                )
                sh 'ls -la'
            }
        }

        stage('Test Docker Access') {
            steps {
                sh 'docker --version'
                sh 'whoami && id'
                sh 'docker info | head -n 15'
            }
        }

        stage('Build Docker Image') {
            steps {
                echo 'Building custom Jenkins image...'
                sh 'ls -la'
                sh 'docker build -t jenkins-anime:latest -f Dockerfile .'
            }
        }
    }

    post {
        success { echo '✅ Pipeline Success!' }
        failure { echo '❌ Pipeline Failed!' }
    }
}