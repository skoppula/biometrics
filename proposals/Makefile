file=skoppula-proposal

pdf: 
		pdflatex --shell-escape ${file}.tex

read:
		evince ${file}.pdf &

clean:
		rm -rf ${file}.log ${file}.aux missfont.log _minted-${file} texput.log

mpc:
		pdflatex --shell-escape ${file}.tex
		rm -rf ${file}.log ${file}.aux missfont.log _minted-${file} texput.log
		evince ${file}.pdf &
