# Run this script to create a properly encoded .env file
with open('.env', 'w', encoding='utf-8') as f:
    f.write('FLASK_SECRET_KEY=1523b284675ee6335f694c12115a63bd\n')
    f.write('# Add other environment variables as needed\n')

print("Successfully created .env file with UTF-8 encoding.")