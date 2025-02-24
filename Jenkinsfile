pipeline {
    agent any

    environment {
        // Nom de l'image Docker sur Docker Hub
        DOCKER_IMAGE = 'willisrunner/mlops:latest'  // Assurez-vous que cette image existe sur Docker Hub
        DOCKER_CREDENTIALS_ID = 'dockerhubcreds'  // ID des identifiants Docker Hub dans Jenkins (configuré dans Jenkins Credentials)
    }

    stages {
        stage('Checkout') {
            steps {
                // Vérifie le code source depuis le dépôt GitHub
                checkout scm
            }
        }

        stage('Pull Docker Image') {
            steps {
                script {
                    // Authentification auprès de Docker Hub et récupération de l'image privée
                    docker.withRegistry('', DOCKER_CREDENTIALS_ID) {
                        def app = docker.image(DOCKER_IMAGE)
                        app.pull()  // S'assure que la dernière version de l'image est récupérée
                    }
                }
            }
        }

        stage('Run Tests') {
            steps {
                script {
                    // Exécute les tests dans l'image Docker
                    docker.withRegistry('', DOCKER_CREDENTIALS_ID) {
                        def app = docker.image(DOCKER_IMAGE)
                        app.inside {
                            // Exécute vos tests avec pytest dans le dossier tests
                            sh 'pytest tests/'  // Adapte ce chemin selon la structure de ton projet
                        }
                    }
                }
            }
        }

        stage('Build and Deploy') {
            steps {
                script {
                    // Build et déploiement avec Docker Compose ou autre outil
                    docker.withRegistry('', DOCKER_CREDENTIALS_ID) {
                        def app = docker.image(DOCKER_IMAGE)
                        app.inside {
                            // Exemple d'exécution de docker-compose pour déployer
                            sh 'docker-compose up -d'  // Assurez-vous que docker-compose est configuré pour ton projet
                        }
                    }
                }
            }
        }

        stage('Notification') {
            steps {
                // Envoi de notification par e-mail après la fin du pipeline
                emailext (
                    subject: "Statut du pipeline CI/CD",
                    body: "Le pipeline a été exécuté avec succès.",
                    to: "NGOAMENYE.LUCAIME@esprit.tn"
                )
            }
        }
    }

    post {
        success {
            echo 'Pipeline exécuté avec succès.'
        }
        failure {
            echo 'Le pipeline a échoué.'
        }
    }
}
