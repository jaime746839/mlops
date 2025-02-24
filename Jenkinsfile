pipeline {
    agent any

    environment {
        // Nom de l'image Docker privée
        DOCKER_IMAGE = 'willisrunner/mlops:latest'  // Assurez-vous de remplacer par votre propre image Docker
        DOCKER_CREDENTIALS_ID = 'docker-hub-creds'  // ID des identifiants Docker Hub dans Jenkins
    }

    stages {
        stage('Checkout') {
            steps {
                // Vérifie le code source depuis votre repository
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
                            // Exécute vos tests avec pytest
                            sh 'pytest tests/'  // Adaptez ce chemin à la structure de votre projet
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
                            sh 'docker-compose up -d'  // Adaptation au besoin de votre projet
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
                    to: "votre_email@example.com"
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
