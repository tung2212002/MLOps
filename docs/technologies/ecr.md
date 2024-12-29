# AWS ECR (Elastic Container Registry)

**AWS ECR** is a fully managed Docker container registry that makes it easy to store, manage, and deploy Docker container images. It integrates well with other AWS services like ECS and EKS for containerized application deployment.

We use ECR in our system to store Docker images of our machine learning models and APIs, which are then deployed in EKS.

### Key Features:
- **Fully Managed**: No need to manage your own container registry.
- **Scalable**: Handles storing and managing any number of images.
- **Secure**: Integrated with AWS IAM for secure access control.

### Example:
```bash
# Push Docker image to ECR
docker tag my_model_api:latest <aws_account_id>.dkr.ecr.<region>.amazonaws.com/my-repo:latest
docker push <aws_account_id>.dkr.ecr.<region>.amazonaws.com/my-repo:latest
