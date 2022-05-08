
## convert the text/attention list to latex code, which will further generates the text heatmap based on attention weights.
import numpy as np

latex_special_token = ["!@#$%^&*()"]

def generate(text_list, attention_list, max_n_elements, latex_file, color='red', rescale_value = False):
	assert(len(text_list) == len(attention_list))
	if rescale_value:
		attention_list = rescale(attention_list)
	word_num = len(text_list)
	text_list = clean_word(text_list)
	with open(latex_file,'w') as f:
		f.write(r'''\documentclass[varwidth]{standalone}
\special{papersize=210mm,297mm}
\usepackage{color}
\usepackage{tcolorbox}
\usepackage{CJK}
\usepackage{adjustbox}
\tcbset{width=0.9\textwidth,boxrule=0pt,colback=red,arc=0pt,auto outer arc,left=0pt,right=0pt,boxsep=5pt}
\begin{document}
\begin{CJK*}{UTF8}{gbsn}'''+'\n')
		string = r'''{\setlength{\fboxsep}{0.4pt}\colorbox{white!0}{\parbox{0.9\textwidth}{'''+"\n"
		for idx in range(word_num): 
			if attention[idx] in max_n_elements:
				string += "\\fbox{\\colorbox{%s!%s}{"%(color, attention_list[idx])+"\\strut " + text_list[idx]+"}} "
			else:
				string += "\\colorbox{%s!%s}{"%(color, attention_list[idx])+"\\strut " + text_list[idx]+"} "
		string += "\n}}}"
		f.write(string+'\n')
		f.write(r'''\end{CJK*}
\end{document}''')

def rescale(input_list):
	the_array = np.asarray(input_list)
	the_max = np.max(the_array)
	the_min = np.min(the_array)
	rescale = (the_array - the_min)/(the_max-the_min)*100
	return rescale.tolist()


def clean_word(word_list):
	new_word_list = []
	for word in word_list:
		for latex_sensitive in ["\\", "%", "&", "^", "#", "_",  "{", "}"]:
			if latex_sensitive in word:
				word = word.replace(latex_sensitive, '\\'+latex_sensitive)
		new_word_list.append(word)
	return new_word_list


if __name__ == '__main__':
	## This is a demo:

	sent = "<sos> a dog participating in a race while wearing the number 6 . <eos>"
	words = sent.split()
	word_num = len(words)
	weights = [3.2011,   3.6839,  -3.3464, -13.6046,  -1.4383,  -1.1158,  -2.7103,
          3.4525, -11.5235,  -8.6178,  -6.0698,  -1.6109,   3.0662,   2.4883]

	scaled_zero = [i - (min(weights)) for i in weights]
	
	attention = [100 - (i/max(scaled_zero))*100 for i in scaled_zero]
	print (attention)

	max_elements_list = attention.copy()
	max_elements_list.sort()
	max_n_elements = max_elements_list[-int(0.2*len(attention)):]

	import random
	# random.seed(42)
	# random.shuffle(attention)
	color = 'red'
	generate(words, attention, max_n_elements, "sample.tex", color)