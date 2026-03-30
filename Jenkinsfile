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
                sh 'docker info | head -n 10'
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
        success { echo '✅ Success!' }
        failure { echo '❌ Failed!' }
    }
}

