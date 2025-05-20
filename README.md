## Construction-Material-Recommendation-System

#Overview

The Construction Material Recommendation System is a web application designed to assist users in selecting construction materials based on project requirements. It supports both admin and user roles, with features like project management, material recommendations, supplier management, and user activity tracking. The system includes email verification during registration to ensure secure account creation, preventing registration completion until the email is verified.

#Key features

User Registration with Email Verification: Users must verify their email using an OTP (One-Time Password) before completing registration.
Role-Based Access: Admins can manage users and suppliers, while users can create projects and view material recommendations.
Material Recommendations: Recommendations are generated based on project criteria like durability, cost, and environmental suitability.
Supplier Database: Manage and view supplier details for sourcing materials.
User Activity Tracking: Admins can monitor user activity for auditing purposes.

#Tech Stack
#Frontend: 
  React (with JSX), Tailwind CSS, Axios for API calls

#Backend: 
  Flask (Python), SQLite (for development), JWT for authentication

#Email Verification: 
  OTP-based email verification (mock email sending in development; requires integration with   an email service in production)

#Dependencies:

Frontend: React 18.2.0, Axios 1.6.7, Tailwind CSS, PapaParse for CSV parsing
Backend: Flask, Fla
