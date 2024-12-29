# Docker

**Docker** is a platform used to develop, ship, and run applications inside lightweight containers. Containers are isolated environments that package all dependencies needed to run a program, making it easy to deploy and scale applications.

In our MLOps pipeline, Docker is used to containerize the entire model training and deployment environment, ensuring consistency across different stages of development and production.

### Key Features:
- **Isolation**: Keeps your application and its dependencies in separate containers.
- **Portability**: Run containers anywhere (locally, in the cloud, etc.).
- **Scalability**: Easily scale applications using Docker orchestration tools like Kubernetes.

We use Docker to create a container image for our model API and deploy it in the cloud.

### Example:
```bash
# Build Docker image
docker build -t my_model_api .

# Run the container locally
docker run -p 8000:8000 my_model_api
