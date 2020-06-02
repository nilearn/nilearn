#! /bin/bash

fa1='http://www.nitrc.org/frs/download.php/1071/NYU_TRT_session1a.tar.gz'
fb1='http://www.nitrc.org/frs/download.php/1072/NYU_TRT_session1b.tar.gz'
fa2='http://www.nitrc.org/frs/download.php/1073/NYU_TRT_session2a.tar.gz'
fb2='http://www.nitrc.org/frs/download.php/1074/NYU_TRT_session2b.tar.gz'
fa3='http://www.nitrc.org/frs/download.php/1075/NYU_TRT_session3a.tar.gz'
fb3='http://www.nitrc.org/frs/download.php/1076/NYU_TRT_session3b.tar.gz'

urls=( $fa1 $fb1 $fa2 $fb2 $fa3 $fb3 )
for url in "${urls[@]}"; do
    printf "URL: <%s>\n" "$url"
    session=$(printf "%s\n" "$url" | sed -n 's/.*\(session..\)\.tar\.gz/\1/p')
    printf "%s\n" "$session"
    ./list_archive_contents.sh "$url" > "archive_contents/nyu_rest/${session}.txt"
done
