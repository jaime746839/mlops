pipeline {
    agent any

    environment {
        // Nom de l'image Docker privée
        DOCKER_IMAGE = 'willisrunner/mlops:latest'  // Assurez-vous de remplacer par votre propre image Docker
        DOCKER_CREDENTIALS_ID = 'docker-hub-creds'  // ID des identifiants Docker Hub dans Jenkins
        GIT_REPO = 'https://github.com/jaime746839/mlops.git'  // Lien vers votre repository Git
    }

    stages {
        // Étape 1 : Cloner le repository Git
        stage('Checkout') {
            steps {
                // Vérifier le code source depuis le repository Git
                git branch: 'master', url: "$GIT_REPO"
            }
        }

        // Étape 2 : Pull de l'image Docker
        stage('Pull Docker Image') {
            steps {
                script {
                    // Authentification auprès de Docker Hub et récupération de l'image privée
                    docker.withRegistry('', DOCKER_CREDENTIALS_ID) {
                        def app = docker.image(DOCKER_IMAGE)
                        app.pull()  // S'assurer que la dernière version de l'image est récupérée
                    }
                }
            }
        }

        // Étape 3 : Exécution des tests
        stage('Run Tests') {
            steps {
                script {
                    // Exécuter les tests dans l'image Docker
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

        // Étape 4 : Build et déploiement avec Docker
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

        // Étape 5 : Notification par e-mail
        stage('Notification') {
            steps {
                // Envoi de notification par e-mail après la fin du pipeline
                emailext (
                    subject: "Statut du pipeline CI/CD",
                    body: "Le pipeline a été exécuté avec succès.",
                    to: "willisrunner811@gmail.com"
                )
            }
        }
    }

    post {
        // Actions à réaliser après l'exécution du pipeline
        success {
            echo 'Pipeline exécuté avec succès.'
        }
        failure {
            echo 'Le pipeline a échoué.'
        }
    }
}
