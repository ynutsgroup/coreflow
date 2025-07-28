#!/bin/bash
# Location: /opt/coreflow/scripts/redisai_infer.sh

REDIS_HOST="${REDIS_HOST:-10.10.10.1}"
REDIS_PORT="${REDIS_PORT:-6380}"
MODEL_NAME="lstm:trading:model"

# Check for Redis password
if [ -z "$REDIS_PASS" ]; then
    echo "❌ Please set REDIS_PASS environment variable first:"
    echo "   export REDIS_PASS='your_password'"
    exit 1
fi

generate_tensor_data() {
    python3 -c "
import numpy as np
np.random.seed(42)
x = np.random.rand(1,60,10).astype(np.float32)
print(' '.join(map(str, x.flatten())))
"
}

run_inference() {
    local tensor_data="$1"
    redis-cli \
        -h "$REDIS_HOST" \
        -p "$REDIS_PORT" \
        -a "$REDIS_PASS" \
        --no-auth-warning <<EOF
AI.TENSORSET input FLOAT 1 60 10 VALUES $tensor_data
AI.MODELRUN $MODEL_NAME INPUTS input OUTPUTS output
AI.TENSORGET output VALUES
EOF
}

echo "🚀 Starting RedisAI Inference Workflow"
TENSOR_DATA=$(generate_tensor_data)

echo -e "\n🔢 Tensor Data Sample:"
echo "$TENSOR_DATA" | awk '{print "First 5 values:", $1, $2, $3, $4, $5, "..."}'

echo -e "\n🧠 Executing Model Prediction..."
RESULT=$(run_inference "$TENSOR_DATA" 2>&1)

if [ $? -eq 0 ]; then
    echo -e "\n✅ Prediction Results:"
    echo "$RESULT" | tail -n +3  # Skip the OK responses
else
    echo -e "\n❌ Execution Failed!"
    echo "Error Details:"
    echo "$RESULT"
    exit 1
fi
