benchmarks_run:
	asv run --show-stderr HASHFILE:hashestobenchmark.txt

benchmarks_view:
	asv publish
	asv preview

clean:
	rm -fr hashestobenchmark_tmp.txt

hashestobenchmark_tmp.txt:
	git rev-parse upstream/main > hashestobenchmark_tmp.txt
	git rev-parse HEAD >> hashestobenchmark_tmp.txt
