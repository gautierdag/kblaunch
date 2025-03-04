#!/bin/sh

#NOTE: Assumes curl and jq are installed

# GitHub OAuth Configuration
CLIENT_ID="01ab8ac9400c4e429b23"  # Official VS Code client ID
SCOPE="repo,read:org,user:email"

# Request device code from GitHub
response=$(curl -s -X POST \
-H "Accept: application/json" \
-d "client_id=$CLIENT_ID" \
-d "scope=$SCOPE" \
https://github.com/login/device/code)

# Parse response
device_code=$(echo "$response" | jq -r .device_code)
user_code=$(echo "$response" | jq -r .user_code)
verification_uri=$(echo "$response" | jq -r .verification_uri)
interval=$(echo "$response" | jq -r .interval)
expires_in=$(echo "$response" | jq -r .expires_in)


echo "device_code: $device_code"
echo "user_code: $user_code"
echo "verification_uri: $verification_uri"
echo "interval: $interval"
echo "expires_in: $expires_in"


# Send Slack message
message="🚨 *VS Code Tunnel Auth Required* 🚨\n\
Please login to GitHub:\n\
1. Visit <$verification_uri|GitHub Verification Page>\n\
2. Enter code: \`$user_code\`\n\
_This code expires in $((expires_in/60)) minutes_"

curl -s -X POST -H 'Content-type: application/json' \
--data "{\"text\":\"$message\"}" \
"$SLACK_WEBHOOK"

# Poll for access token
while [ $expires_in -gt 0 ]; do
    sleep $interval
    token_response=$(curl -s -X POST \
        -H "Accept: application/json" \
        -d "client_id=$CLIENT_ID" \
        -d "device_code=$device_code" \
        -d "grant_type=urn:ietf:params:oauth:grant-type:device_code" \
        https://github.com/login/oauth/access_token)

    if echo "$token_response" | jq -e '.access_token' >/dev/null; then
        access_token=$(echo "$token_response" | jq -r .access_token)
        break
    fi
    expires_in=$((expires_in - interval))
done


# Authenticate VS Code CLI
code tunnel user login --provider github --access-token "$access_token"

# Clip pod name to <20 characters
TUNNEL_NAME=$(echo $POD_NAME | cut -c 1-20)

# Start tunnel
code tunnel --name $TUNNEL_NAME --accept-server-license-terms