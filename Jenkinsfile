pipeline {
    agent any

    stages {
        stage('Checkout') {
            steps {
                echo 'Checking out code from GitHub...'
                checkout scmGit(
                    branches: [[name: '*/main']], 
                    extensions: [], 
                    userRemoteConfigs: [[
                        credentialsId: 'github-token', 
                        url: 'https://github.com/sankarayinala/MLOPS-Experiment02.git'
                    ]]
                )
            }
        }

        stage('Build Docker Image') {
            steps {
                echo 'Building custom Jenkins image...'
                sh 'docker build -t jenkins-anime:latest .'
            }
        }

        // Add your ML pipeline stages here later (e.g. test, train model, etc.)
    }

    post {
        always {
            echo 'Pipeline finished!'
        }
        success {
            echo '✅ Build successful!'
        }
        failure {
            echo '❌ Pipeline failed!'
        }
    }
}