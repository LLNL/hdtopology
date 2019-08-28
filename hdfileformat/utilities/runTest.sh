red='\033[0;31m'
NC='\033[0m' # No Color

for testCase in unit_test_metaInfo_write unit_test_metaInfo_read unit_test_metaInfo_rewrite unit_test_metaInfo_read
do
  if ./$testCase>/dev/null; then
    echo "$testCase Success"
  else
    echo "${red} $testCase Fail ${NC}"
  fi
done
