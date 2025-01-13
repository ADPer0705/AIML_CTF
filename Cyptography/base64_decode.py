import base64

sub1 = 'TFJHTXtY'
sub2 = 'S2IzWHkx'
sub3 = 'dG1fWm4z'
sub4 = 'X1Zlem4w'
sub5 = 'dF9JMGoz'
sub6 = 'X0l4NGlx'
sub7 = 'M2pfT3oh'
sub8 = 'fQ=='

# Decode each substring and combine them sequentially
decoded_bytes = base64.b64decode(sub1) + base64.b64decode(sub2) + base64.b64decode(sub3) + \
                base64.b64decode(sub4) + base64.b64decode(sub5) + base64.b64decode(sub6) + \
                base64.b64decode(sub7) + base64.b64decode(sub8)
decoded_string = decoded_bytes.decode('utf-8')

print("Decoded string:", decoded_string)