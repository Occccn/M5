{\rtf1\ansi\ansicpg932\cocoartf2512
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fnil\fcharset0 Menlo-Regular;\f1\froman\fcharset0 Times-Roman;}
{\colortbl;\red255\green255\blue255;\red140\green155\blue158;\red220\green220\blue220;\red230\green203\blue56;
\red55\green169\blue206;\red0\green0\blue233;}
{\*\expandedcolortbl;;\cssrgb\c61569\c67059\c68235;\cssrgb\c89020\c89020\c89020;\cssrgb\c92157\c82353\c27843;
\cssrgb\c25490\c71765\c84314;\cssrgb\c0\c0\c93333;}
\paperw11900\paperh16840\margl1440\margr1440\vieww10800\viewh8400\viewkind0
\deftab720
\pard\pardeftab720\sl500\partightenfactor0

\f0\fs28\fsmilli14400 \cf2 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 #!/bin/bash\cf3 \strokec3 \
\
\pard\pardeftab720\sl500\partightenfactor0
\cf4 \strokec4 for \cf3 \strokec3 i \cf4 \strokec4 in\cf3 \strokec3  \cf5 \strokec5 `\cf3 \strokec3 seq 0 12\cf5 \strokec5 `\cf3 \strokec3 \
\cf4 \strokec4 do\
  \cf3 \strokec3 echo \cf5 \strokec5 "[\cf3 \strokec3 $i\cf5 \strokec5 ]"\cf3 \strokec3  \cf5 \strokec5 `\cf3 \strokec3  date \cf5 \strokec5 '+%y/%m/%d %H:%M:%S'`\cf3 \strokec3  \cf5 \strokec5 "connected."\cf3 \strokec3 \
  open <<{\field{\*\fldinst{HYPERLINK "https://colab.research.google.com/drive/1h80COehkiT2E9SJY-1sU_KxvUkj_3k6s"}}{\fldrslt 
\f1\fs24 \cf6 \ul \ulc6 \strokec6 https://colab.research.google.com/drive/1h80COehkiT2E9SJY-1sU_KxvUkj_3k6s}}>>\
  sleep 60\
\cf4 \strokec4 done\cf3 \strokec3 \
}