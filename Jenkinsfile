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
                echo 'Building Jenkins DinD Docker Image...'
                sh '''
                    docker build -t jenkins-anime .
                '''
            }
        }

        stage('Run Container') {
            steps {
                echo 'Stopping old container if exists...'
                sh '''
                    docker rm -f jenkins-dind || true
                '''
                
                echo 'Running new Jenkins DinD container...'
                sh '''
                    docker run -d \
                      --name jenkins-dind \
                      --privileged \
                      -p 8080:8080 \
                      -p 50000:50000 \
                      -v jenkins_home:/var/jenkins_home \
                      jenkins-anime
                '''
            }
        }
    }
    
    post {
        always {
            echo 'Pipeline finished!'
        }
        success {
            echo '✅ Build and deployment successful!'
        }
        failure {
            echo '❌ Pipeline failed!'
        }
    }
}