Note:
1.Sagemaker Permission
User needs to give the Sagemaker role lambda function permissions.
It is the Sagemaker role, not the user IAM role.

2.SNS Permission
Authorize the IAM role (according to the error information), as the production role may be assume
so it's dynamic. In this case, SNS service subscription may not work.