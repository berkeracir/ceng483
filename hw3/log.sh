x=1
result=""
for i in $(grep -i "Validation-Loss" $1 | cut -d'|' -f3 | cut -d':' -f2) ; do 
    result="$result ($x,$i)"
    x=$((x+1))
done

echo $result
