"""Run a language-model experiment: python run.py JOB [arguments].

Jobs: prep, eval, aux, wsx, probe, train, qwennan, nativecheck, gradcheck, trainseed, judge.
Local files are used by default. Cloud synchronization requires RX_LOCAL=0 and S3_BUCKET."""
import os
import sys
import time
import traceback

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rx import common  # noqa: E402

JOB = os.environ.get("S3_PREFIX", "local")


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        raise SystemExit(__doc__)
    job, args = sys.argv[1], sys.argv[2:]
    common.log("run", "job=%s box=%s args=%s" % (job, JOB, args))
    common.write_json(common.RESULTS / "START.json",
                      dict(job=job, args=args, box=JOB, env=common.env_info(),
                           utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))
    try:
        if job == "prep":
            from rx import prep
            m = args[args.index("--model") + 1]
            if m == "llama-3.1-70b" and "--archived-mean" not in args:
                d = common.WORK / "archived_70b"
                for f in ("mean.npy", "pca_50000.npz", "meta.json"):
                    common.s3_get("rx-modal-70b/acts/llama-3.1-70b_wikitext/" + f, d / f)
                args += ["--archived-mean", str(d)]
            out = prep.main(args)
            payload = dict(gate3=out.get("gate3", {}).get("status"))
        elif job == "eval":
            from rx import evaljob
            payload = evaljob.main(args) or {}
        elif job == "aux":
            from rx import auxjob
            payload = auxjob.main(args) or {}
        elif job == "wsx":
            from rx import wsxjob
            payload = wsxjob.main(args) or {}
        elif job == "probe":
            from rx import probejob
            payload = probejob.main(args) or {}
        elif job == "train":
            from rx import trainjob
            payload = trainjob.main(args) or {}
        elif job == "qwennan":
            from rx import qwennan
            payload = qwennan.main(args) or {}
        elif job == "nativecheck":
            from rx import nativecheck
            payload = nativecheck.main(args) or {}
        elif job == "gradcheck":
            from rx import gradcheck
            payload = gradcheck.main(args) or {}
        elif job == "trainseed":
            from rx import trainseed
            payload = trainseed.main(args) or {}
        elif job == "judge":
            from rx import judge
            payload = judge.main(args) or {}
        else:
            raise SystemExit("unknown job " + job)
        common.mark_done(JOB, payload)
        common.log("run", "DONE")
    except Exception as e:
        tb = traceback.format_exc()
        print(tb, flush=True)
        common.mark_failed(JOB, tb)
        sys.exit(1)


if __name__ == "__main__":
    main()
