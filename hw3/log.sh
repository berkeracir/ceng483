x=5
result=""
for i in $(grep -i "Validation-Loss" $1 | cut -d':' -f2) ; do 
    result="$result ($x,$i)"
    x=$((x+5))
done

echo $result
