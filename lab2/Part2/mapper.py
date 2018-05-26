#!/usr/bin/ python3.4
import sys
import re

def read_input(file):
    for line in file:
        # split the line into words
        yield line.split()

def main(separator='\t'):
    # input comes from STDIN (standard input)
    data = read_input(sys.stdin)
    stopwords = ['a','an','the','and','is','are','was','were','what','them','had','some','ca',
                 'why','when','where','who','whose','which','that','off','ever','many','ve',
                 'those','this','those','but','so','thus','again','therefore','its','both',
				     'like','in','on','up','down','under','over','i','we','they','while','okay',
				     'he','them','their','there','us','of','you','your','us','our','mine','mr',
                 'such','am','to','too','for','from','since','until','between','she','own',
                 'my','not','if','as', 'well','youre','hadnt','havent','wont','q','se','ok',
                 'very','have','it','be','been','has','having','his', 'her','never','above',
                 'should','would', 'could','just', 'about','do','doing','does','did','la','ha'
                 'go','going','goes','being','with', 'yes', 'no','how','before','than','d',
                 'after','any','here','out','now','then','got','into','all','cant','or','ya',
                 'despite','beyond','further','wanna', 'want','gonna','isnt', 'at','also','lo',
                 'because','due','heres','try','said','says','will','shall','link','asked',
                 'more','less','often','lol','maybe','perhaps','quite','even','him','by','n',
                 'among','can','may','most','took','during','me','told','might','hi','es','l',
                 'theyll','use','u','whats','couldnt','wouldnt','see','im','dont','x','de',
                 'doesnt','shouldnt', 'hes','thats','let','lets','get','gets','en','co','k',
                 'whats','s','say','via','youll','wed','theyd','youd','w','m','hey','hello',
                 'youve','theyve','weve','theyd','youd','ive','were','ill','yet','b','rt',
                 'id','o','r','z','um','em','seen','didnt','r','e','t','c','y','only','v',
                 'arent','werent','hasnt','mostly','much','ago','wasnt','aint','nope','p',
                 'll','ja','al','el','gt','cs','si','didn','re','f','fo','j','ni','tr','il']
    for words in data:
        x = ' '.join(words)
        x = re.sub(r'http+s*\:+\/\/[a-zA-Z\.\/0-9]+ *', '', x) 
        x = re.sub(r' {1}[a-zA-Z0-9]+…$','',x)
        x = re.sub(r'[\)\( \/\.-][0-9]+[ \)\/\.\(-]',' ',x)
        x = x.replace('.',' ')
        x = re.sub(r'RT @\S+', '', x)
        x = re.sub(r'@\S+', '', x)
        x = x.lower()
        x = x.replace("'s","")
        x = re.sub(r"[^a-zA-Z0-9 ]+", ' ', x)
        x = re.sub(r'[ ][0-9]+ ','',x)
        coll = x.split()
        coll = [w for w in coll if not w in stopwords]
        # write the results to STDOUT (standard output);
        # what we output here will be the input for the
        # Reduce step, i.e. the input for reducer.py
        #
        # tab-delimited; the trivial word count is 1
        for word in coll:
            print('%s%s%d' % (word, separator, 1))
if __name__ == "__main__":
    main()