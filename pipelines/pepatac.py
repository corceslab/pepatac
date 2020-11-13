#!/usr/bin/env python
"""
PEPATAC - ATACseq pipeline
"""

__author__ = ["Jin Xu", "Nathan Sheffield", "Jason Smith"]
__email__ = "jasonsmith@virginia.edu"
__version__ = "0.9.9"


from argparse import ArgumentParser
import os
import re
import sys
import tempfile
import pypiper
import pandas as pd
import numpy as np
from pypiper import build_command
from refgenconf import RefGenConf as RGC, select_genome_config

TOOLS_FOLDER = "tools"
ANNO_FOLDER = "anno"
ALIGNERS = ["bowtie2", "bwa"]
PEAK_CALLERS = ["fseq", "genrich", "hmmratac", "homer", "macs2"]
PEAK_TYPES = [ "fixed", "variable"]
DEDUPLICATORS = ["picard", "samblaster", "samtools"]
TRIMMERS = ["trimmomatic", "pyadapt", "skewer"]
GENOME_IDX_KEY = "bowtie2_index"


def parse_arguments():
    """
    Parse command-line arguments passed to the pipeline.
    """
    # Argument Parsing from yaml file
    ###########################################################################
    parser = ArgumentParser(description='PEPATAC version ' + __version__)
    parser = pypiper.add_pypiper_args(parser, groups=
        ['pypiper', 'looper', 'ngs'],
        required=["input", "genome", "sample-name", "output-parent"])

    # Pipeline-specific arguments
    parser.add_argument("--aligner", dest="aligner", type=str.lower,
                        default="bowtie2", choices=ALIGNERS,
                        help="Name of read aligner")
                        
    parser.add_argument("--peak-caller", dest="peak_caller",
                        default="macs2", choices=PEAK_CALLERS,
                        help="Name of peak caller")

    parser.add_argument("-gs", "--genome-size", default="2.7e9", type=str.lower,
                        help="Effective genome size. It can be 1.0e+9 "
                        "or 1000000000: e.g. human (2.7e9), mouse (1.87e9), "
                        "C. elegans (9e7), fruitfly (1.2e8). Default:2.7e9")

    parser.add_argument("--trimmer", dest="trimmer", type=str.lower,
                        default="skewer", choices=TRIMMERS,
                        help="Name of read trimming program")

    parser.add_argument("--prealignments", default=[], type=str,
                        nargs="+",
                        help="Space-delimited list of reference genomes to "
                             "align to before primary alignment.")

    parser.add_argument("--deduplicator", dest="deduplicator", type=str.lower,
                        default="samblaster", choices=DEDUPLICATORS,
                        help="Name of deduplicator program")

    parser.add_argument("--TSS-name", default=None,
                        dest="TSS_name", type=str.lower,
                        help="Path to TSS annotation file.")

    parser.add_argument("--blacklist", default=None,
                        dest="blacklist", type=str.lower,
                        help="Path to genomic region blacklist file")

    parser.add_argument("--anno-name", default=None,
                        dest="anno_name", type=str.lower,
                        help="Path to reference annotation file (BED format) for calculating FRiF")

    parser.add_argument("--peak-type", default="fixed",
                        dest="peak_type", choices=PEAK_TYPES, type=str.lower,
                        help="Call variable or fixed width peaks.\n"
                             "Fixed width requires MACS2.")

    parser.add_argument("--extend", default=250,
                        dest="extend", type=int,
                        help="How far to extend fixed width peaks up and "
                             "downstream.")

    parser.add_argument("--frip-ref-peaks", default=None,
                        dest="frip_ref_peaks", type=str.lower,
                        help="Path to reference peak set (BED format) for calculating FRiP")

    parser.add_argument("--motif", action='store_true',
                        dest="motif",
                        help="Perform motif enrichment analysis")

    parser.add_argument("--sob", action='store_true',
                        dest="sob", default=False,
                        help="Use seqOutBias to produce signal tracks, "
                             "incorporate mappability information, "
                             "and account for Tn5 bias.")
    
    parser.add_argument("--no-scale", action='store_true',
                        dest="no_scale", default=False,
                        help="Do not scale signal tracks: "
                             "Default is to scale by read count.\n"
                             "If using seqOutBias, scales by the expected/"
                             "observed cut frequency.")

    parser.add_argument("--prioritize", action='store_true', default=False,
                        dest="prioritize",
                        help="Plot cFRiF/FRiF using mutually exclusive priority"
                             " ranked features based on the order of feature"
                             " appearance in the feature annotation asset.")

    parser.add_argument("--keep", action='store_true',
                        dest="keep",
                        help="Enable this flag to keep prealignment BAM files")
                    
    parser.add_argument("--noFIFO", action='store_true',
                        dest="no_fifo",
                        help="Do NOT use named pipes during prealignments")

    parser.add_argument("--lite", dest="lite", action='store_true',
                        help="Only keep minimal, essential output to conserve "
                             "disk space.")

    parser.add_argument("--skipqc", dest="skipqc", action='store_true',
                        help="Skip FastQC. Useful for bugs in FastQC "
                             "that appear with some sequence read files.")

    parser.add_argument("-V", "--version", action="version",
                        version="%(prog)s {v}".format(v=__version__))

    args = parser.parse_args()

    # TODO: determine if it's safe to handle this requirement with argparse.
    # It may be that communication between pypiper and a pipeline via
    # the pipeline interface (and/or) looper, and how the partial argument
    # parsing is handled, that makes this more favorable.
    if not args.input:
        parser.print_help()
        raise SystemExit

    return args


def report_message(pm, report_file, message, annotation=None):
    """
    Writes a string to provided file in a safe way.
    
    :param PipelineManager pm: a pypiper PipelineManager object
    :param str report_file: name of the output file
    :param str message: string to write to the output file
    :param str annotation: By default, the message will be annotated with the
        pipeline name, so you can tell which pipeline records which stats.
        If you want, you can change this; use annotation='shared' if you
        need the stat to be used by another pipeline (using get_stat()).
    """
    # Default annotation is current pipeline name.
    annotation = str(annotation or pm.name)

    message = str(message).strip()
    
    message = "{message}\t{annotation}".format(
        message=message, annotation=annotation)

    # Just to be extra careful, let's lock the file while we we write
    # in case multiple pipelines write to the same file.
    pm._safe_write_to_file(report_file, message)


def calc_frip(bamfile, peakfile, frip_func, pipeline_manager,
              aligned_reads_key="Aligned_reads"):
    """
    Calculate the fraction of reads in peaks (FRIP).

    Use the given function and data from an aligned reads file and a called
    peaks file, along with a PipelineManager, to calculate FRIP.

    :param str peakfile: path to called peaks file
    :param callable frip_func: how to calculate the fraction of reads in peaks;
        this must accept the path to the aligned reads file and the path to
        the called peaks file as arguments.
    :param str bamfile: path to aligned reads file
    :param pypiper.PipelineManager pipeline_manager: the PipelineManager in use
        for the pipeline calling this function
    :param str aligned_reads_key: name of the key from a stats (key-value) file
        to use to fetch the count of aligned reads
    :return float: fraction of reads in peaks
    """
    frip_cmd = frip_func(bamfile, peakfile)
    num_peak_reads = pipeline_manager.checkprint(frip_cmd)
    num_aligned_reads = pipeline_manager.get_stat(aligned_reads_key)
    print(num_aligned_reads, num_peak_reads)
    return float(num_peak_reads) / float(num_aligned_reads)


def _align(args, tools, paired, useFIFO, unmap_fq1, unmap_fq2,
           assembly_identifier, assembly, outfolder,
           aligndir=None, bt2_opts_txt=None, bwa_opts_txt=None):
    """
    A helper function to run alignments in series, so you can run one alignment
    followed by another; this is useful for successive decoy alignments.

    :param argparse.Namespace args: binding between option name and argument,
        e.g. from parsing command-line options
    :param looper.models.AttributeDict tools: binding between tool name and
        value, e.g. for tools/resources used by the pipeline
    :param bool paired: if True, use paired-end alignment
    :param bool useFIFO: if True, use named pipe instead of file creation
    :param str unmap_fq1: path to unmapped read1 FASTQ file
    :param str unmap_fq2: path to unmapped read2 FASTQ file
    :param str assembly_identifier: text identifying a genome assembly for the
        pipeline
    :param str assembly: assembly-specific folder (index, etc.)
    :param str outfolder: path to output directory for the pipeline
    :param str aligndir: name of folder for temporary output
    :param str bt2_opts_txt: command-line text for bowtie2 options
    :param str bwa_opts_txt: command-line text for bwa options
    :return (str, str): pair (R1, R2) of paths to FASTQ files
    """
    pm.timestamp("### Map to " + assembly_identifier)
    if not aligndir:
        align_subdir = "aligned_{}_{}".format(args.genome_assembly,
                                              assembly_identifier)
        sub_outdir = os.path.join(outfolder, align_subdir)
    else:
        sub_outdir = os.path.join(outfolder, aligndir)

    ngstk.make_dir(sub_outdir)
    bamname = "{}_{}.bam".format(args.sample_name, assembly_identifier)
    all_mapped_bam = os.path.join(sub_outdir, 
        args.sample_name + "_" + assembly_identifier + "_all.bam")
    mapped_bam = os.path.join(sub_outdir, bamname)
    unmapped_bam = os.path.join(sub_outdir, 
        args.sample_name + "_" + assembly_identifier + "_unmapped.bam")
    summary_name = "{}_{}_bt_aln_summary.log".format(args.sample_name,
                                                     assembly_identifier)
    summary_file = os.path.join(sub_outdir, summary_name)

    out_fastq_pre = os.path.join(
        sub_outdir, args.sample_name + "_" + assembly_identifier)

    out_fastq_r1    = out_fastq_pre + '_unmap_R1.fq'
    out_fastq_r1_gz = out_fastq_r1  + '.gz'

    out_fastq_r2    = out_fastq_pre + '_unmap_R2.fq'
    out_fastq_r2_gz = out_fastq_r2  + '.gz'

    if (useFIFO and
            paired and not
            args.keep and not
            args.aligner.lower() == "bwa"):
        out_fastq_tmp = os.path.join(sub_outdir,
                assembly_identifier + "_bt2")
        cmd = "mkfifo " + out_fastq_tmp

        if os.path.exists(out_fastq_tmp):
            os.remove(out_fastq_tmp)
        pm.run(cmd, out_fastq_tmp)
    else:
        out_fastq_tmp    = out_fastq_pre + '_unmap.fq'
        out_fastq_tmp_gz = out_fastq_tmp + ".gz"

    filter_pair = build_command([tools.perl,
        tool_path("filter_paired_fq.pl"), out_fastq_tmp,
        unmap_fq1, unmap_fq2, out_fastq_r1, out_fastq_r2])

    # samtools sort needs a temporary directory
    tempdir = tempfile.mkdtemp(dir=sub_outdir)
    os.chmod(tempdir, 0o771)
    pm.clean_add(tempdir)

    if args.aligner == "bwa":
        if not bwa_opts_txt:
            # Default options
            bwa_opts_txt = "-M"
            bwa_opts_txt += " -SP" # Treat as single end no matter source
            bwa_opts_txt += " -r 3" # Increase speed at cost of accuracy
        # build bwa command
        cmd1 = tools.bwa + " mem -t " + str(pm.cores)
        cmd1 += " " + bwa_opts_txt
        cmd1 += " " + assembly
        cmd1 += " " + unmap_fq1
        cmd1 += " | " + tools.samtools + " view -bS - -@ 1"  # convert to bam
        cmd1 += " | " + tools.samtools + " sort - -@ 1"      # sort output
        cmd1 += " -T " + tempdir
        cmd1 += " -o " + all_mapped_bam
        pm.clean_add(all_mapped_bam)

        # get unmapped reads
        cmd2 = tools.samtools + " view -bS -f 4 -@ " + str(pm.cores)
        cmd2 += " " +  all_mapped_bam
        cmd2 += " > " + unmapped_bam

        # get mapped reads (don't remove if args.keep)
        cmd3 = tools.samtools + " view -bS -F 4 -@ " + str(pm.cores)
        cmd3 += " " +  all_mapped_bam
        cmd3 += " > " + mapped_bam
        if not args.keep:
            pm.clean_add(mapped_bam)

        # Convert bam to fastq for bwa requirement
        cmd4 = tools.bedtools + " bamtofastq "
        cmd4 += " -i " + unmapped_bam
        cmd4 += " -fq " + out_fastq_tmp
        pm.clean_add(unmapped_bam)

        if paired:
            pm.run([cmd1, cmd2, cmd3, cmd4, filter_pair], out_fastq_r2_gz)
        else:
            if args.keep:
                pm.run(cmd, mapped_bam)
            else:
                pm.run(cmd, out_fastq_tmp_gz)

        cmd = tools.samtools + " view -c " + mapped_bam
        align_exact = pm.checkprint(cmd)       
    else:
        if not bt2_opts_txt:
            # Default options
            bt2_opts_txt = "-k 1"  # Return only 1 alignment
            bt2_opts_txt += " -D 20 -R 3 -N 1 -L 20 -i S,1,0.50"           

        # Build bowtie2 command
        cmd = "(" + tools.bowtie2 + " -p " + str(pm.cores)
        cmd += " " + bt2_opts_txt
        cmd += " -x " + assembly
        cmd += " --rg-id " + args.sample_name
        cmd += " -U " + unmap_fq1
        cmd += " --un " + out_fastq_tmp
        if args.keep: #  or not paired
            #cmd += " --un-gz " + out_fastq_bt2 
            # Drop this for paired...repairing with filter_paired_fq.pl
            # In this samtools sort command we print to stdout and then use > to
            # redirect instead of  `+ " -o " + mapped_bam` because then samtools
            # uses a random temp file, so it won't choke if the job gets
            # interrupted and restarted at this step.
            cmd += " | " + tools.samtools + " view -bS - -@ 1"  # convert to bam
            cmd += " | " + tools.samtools + " sort - -@ 1"      # sort output
            cmd += " -T " + tempdir
            cmd += " -o " + mapped_bam
        else:
            cmd += " > /dev/null"
        cmd += ") 2>" + summary_file

        if paired:
            if args.keep or not useFIFO:
                pm.run([cmd, filter_pair], mapped_bam)
            else:
                pm.wait = False
                pm.run(filter_pair, [summary_file, out_fastq_r2_gz])
                pm.wait = True
                pm.run(cmd, [summary_file, out_fastq_r2_gz])
        else:
            if args.keep:
                pm.run(cmd, mapped_bam)
            else:
                # TODO: switch to this once filter_paired_fq works with SE
                #pm.run(cmd2, summary_file)
                #pm.run(cmd1, out_fastq_r1)
                pm.run(cmd, out_fastq_tmp_gz)
        cmd = ("grep 'aligned exactly 1 time' " + summary_file +
               " | awk '{print $1}'")
        align_exact = pm.checkprint(cmd)

    pm.clean_add(out_fastq_tmp)
 
    # report aligned reads
    if align_exact:
        ar = float(align_exact)*2
    else:
        ar = 0
    pm.report_result("Aligned_reads_" + assembly_identifier, ar)
    try:
        # wrapped in try block in case Trimmed_reads is not reported in this
        # pipeline.
        tr = float(pm.get_stat("Trimmed_reads"))
    except:
        print("Trimmed reads is not reported.")
    else:
        res_key = "Alignment_rate_" + assembly_identifier
        pm.report_result(res_key, round(float(ar) * 100 / float(tr), 2))
    
    if paired:
        unmap_fq1 = out_fastq_r1
        unmap_fq2 = out_fastq_r2
    else:
        # Use alternate once filter_paired_fq is working with SE
        #unmap_fq1 = out_fastq_r1
        unmap_fq1 = out_fastq_tmp
        unmap_fq2 = ""

    return unmap_fq1, unmap_fq2


def tool_path(tool_name):
    """
    Return the path to a tool used by this pipeline.

    :param str tool_name: name of the tool (e.g., a script filename)
    :return str: real, absolute path to tool (expansion and symlink resolution)
    """

    return os.path.join(os.path.dirname(os.path.dirname(__file__)),
                        TOOLS_FOLDER, tool_name)


def check_commands(commands, ignore=''):
    """
    Check if command(s) can be called

    :param attributedict commands: dictionary of commands to check
    :param list ignore: list of commands that are optional and can be ignored
    """

    # Use `command` to see if command is callable, store exit code
    is_callable = True
    uncallable = []
    for name, command in commands.items():
        if command not in ignore:
            # if an environment variable is not expanded it means it points to
            # an uncallable command
            if '$' in command:
                # try to expand
                command = os.path.expandvars(os.path.expanduser(command))
                if not os.path.exists(command):
                    uncallable.append(command)

            # if a command is a java file, modify the command
            if '.jar' in command:
                command = "java -jar " + command

            code = os.system("command -v {0} >/dev/null 2>&1 || {{ exit 1; }}".format(command))
            # If exit code is not 0, track which command failed
            #print("{} code {}".format(command, code))  # DEBUG
            if code != 0:
                uncallable.append(command)
                is_callable = False
    if is_callable:
        return True
    else:
        print("The following required tool(s) are not callable: {0}".format(' '.join(uncallable)))
        return False


def _add_resources(args, res, asset_dict=None):
    """
    Add additional resources needed for pipeline.

    :param argparse.Namespace args: binding between option name and argument,
        e.g. from parsing command-line options
    :param pm.config.resources res: pipeline manager resources list
    :param asset_dict list: list of dictionary of assets to add
    """

    rgc = RGC(select_genome_config(res.get("genome_config")))

    key_errors = []
    exist_errors = []
    required_list = []

    # Check that bowtie2/bwa indicies exist for specified prealignments
    for reference in args.prealignments:
        for asset in [GENOME_IDX_KEY]:
            try:
                res[asset] = rgc.seek(reference, asset)
            except KeyError:
                err_msg = "{} for {} is missing from REFGENIE config file."
                pm.fail_pipeline(KeyError(err_msg.format(asset, reference)))
            except:
                err_msg = "{} for {} does not exist."
                pm.fail_pipeline(IOError(err_msg.format(asset, reference)))

    # Check specified assets
    if not asset_dict:
        return res, rgc
    else:
        for item in asset_dict:
            pm.debug("item: {}".format(item))  # DEBUG
            asset = item["asset_name"]
            seek_key = item["seek_key"] or item["asset_name"]
            tag = item["tag_name"] or "default"
            arg = item["arg"]
            user_arg = item["user_arg"]
            req = item["required"]

            if arg and hasattr(args, arg) and getattr(args, arg):
                res[seek_key] = os.path.abspath(getattr(args, arg))
            else:
                try:
                    pm.debug("{} - {}.{}:{}".format(args.genome_assembly,
                                                    asset,
                                                    seek_key,
                                                    tag))  # DEBUG
                    res[seek_key] = rgc.seek(args.genome_assembly,
                                             asset_name=str(asset),
                                             tag_name=str(tag),
                                             seek_key=str(seek_key))
                except KeyError:
                    key_errors.append(item)
                    if req:
                        required_list.append(item)
                except:
                    exist_errors.append(item)
                    if req:
                        required_list.append(item)

        if len(key_errors) > 0 or len(exist_errors) > 0:
            pm.info("Some assets are not found. You can update your REFGENIE "
                    "config file or point directly to the file using the noted "
                    "command-line arguments:")

        if len(key_errors) > 0:
            if required_list:
                err_msg = "Required assets missing from REFGENIE config file: {}"
                pm.fail_pipeline(IOError(err_msg.format(", ".join(["{asset_name}.{seek_key}:{tag_name}".format(**x) for x in required_list]))))
            else:
                warning_msg = "Optional assets missing from REFGENIE config file: {}"
                pm.info(warning_msg.format(", ".join(["{asset_name}.{seek_key}:{tag_name}".format(**x) for x in key_errors])))

        if len(exist_errors) > 0:
            if required_list:
                err_msg = "Required assets not existing: {}"
                pm.fail_pipeline(IOError(err_msg.format(", ".join(["{asset_name}.{seek_key}:{tag_name} (--{user_arg})".format(**x) for x in required_list]))))
            else:
                warning_msg = "Optional assets not existing: {}"
                pm.info(warning_msg.format(", ".join(["{asset_name}.{seek_key}:{tag_name} (--{user_arg})".format(**x) for x in exist_errors])))

        return res, rgc


################################################################################
#                                 Pipeline MAIN                                #
################################################################################
def main():
    """
    Main pipeline process.
    """

    args = parse_arguments()

    args.paired_end = args.single_or_paired.lower() == "paired"

    # Initialize, creating global PipelineManager and NGSTk instance for
    # access in ancillary functions outside of main().
    outfolder = os.path.abspath(
        os.path.join(args.output_parent, args.sample_name))
    global pm
    pm = pypiper.PipelineManager(
        name="PEPATAC", outfolder=outfolder, args=args, version=__version__)
    global ngstk
    ngstk = pypiper.NGSTk(pm=pm)

    # Convenience alias
    tools = pm.config.tools
    param = pm.config.parameters
    param.outfolder = outfolder
    res = pm.config.resources

    ############################################################################
    #                Confirm required tools are all callable                   #
    ############################################################################
    opt_tools = ["fseq", "genrich", "${HMMRATAC}", "${PICARD}",
                 "${TRIMMOMATIC}", "pyadapt", "findMotifsGenome.pl",
                 "findPeaks", "seqOutBias", "bigWigMerge", "bedGraphToBigWig",
                 "pigz", "bwa"]

    # Confirm compatible peak calling settings
    if args.peak_type == "fixed" and not args.peak_caller == "macs2":
        err_msg = ("Must use MACS2 with `--peak-type fixed` width peaks. " +
                   "Either change the " +
                   "`--peak-caller {}` or ".format(PEAK_CALLERS) +
                   "use `--peak-type variable`.")
        pm.fail_pipeline(RuntimeError(err_msg))

    # If using optional tools, remove those from the skipped checks
    if args.aligner == "bwa":
        if 'bwa' in opt_tools: opt_tools.remove('bwa')

    if args.trimmer == "trimmomatic":
        if '${TRIMMOMATIC}' in opt_tools: opt_tools.remove('${TRIMMOMATIC}')
        if not ngstk.check_command(tools.trimmomatic):
            err_msg = ("Unable to call trimmomatic as specified in the " +
                       "pipelines/pepatac.yaml configuration file: " +
                       "".join(str(tools.trimmomatic)))
            pm.fail_pipeline(RuntimeError(err_msg))

    if args.trimmer == "pyadapt":
        if 'pyadapt' in opt_tools: opt_tools.remove('pyadapt')

    if args.deduplicator == "picard":
        if '${PICARD}' in opt_tools: opt_tools.remove('${PICARD}')
        if not ngstk.check_command(tools.picard):
            err_msg = ("Unable to call picard as specified in the " +
                       "pipelines/pepatac.yaml configuration file: " +
                       "".join(str(tools.picard)))
            pm.fail_pipeline(RuntimeError(err_msg))

    if args.peak_caller == "fseq":
        if 'fseq' in opt_tools: opt_tools.remove('fseq')

    if args.peak_caller == "genrich":
        if 'genrich' in opt_tools: opt_tools.remove('genrich')

    if args.peak_caller == "hmmratac":
        if '${HMMRATAC}' in opt_tools: opt_tools.remove('${HMMRATAC}')
        if not ngstk.check_command(tools.hmmratac):
            err_msg = ("Unable to call hmmratac as specified in the " +
                       "pipelines/pepatac.yaml configuration file: " +
                       "".join(str(tools.hmmratac)))
            pm.fail_pipeline(RuntimeError(err_msg))
    
    if args.peak_caller == "homer":
        if 'findPeaks' in opt_tools: opt_tools.remove('findPeaks')

    if args.sob:
        if 'seqOutBias' in opt_tools: opt_tools.remove('seqOutBias')
        if 'bigWigMerge' in opt_tools: opt_tools.remove('bigWigMerge')
        if 'bedGraphToBigWig' in opt_tools: opt_tools.remove('bedGraphToBigWig')

    if args.motif:
        if 'findMotifsGenome.pl' in opt_tools: opt_tools.remove('findMotifsGenome.pl')

    # Check that the required tools are callable by the pipeline
    tool_list = [v for k,v in tools.items()]  # extract tool list
    if args.peak_caller == "homer":
        tool_list.append('makeTagDirectory')
        tool_list.append('pos2bed.pl')
    pm.debug(tool_list)  # DEBUG
    tool_list = [t.replace('seqoutbias', 'seqOutBias') for t in tool_list]
    tool_list = dict((t,t) for t in tool_list)  # convert back to dict

    if not check_commands(tool_list, opt_tools):
        err_msg = "Missing required tools. See message above."
        pm.fail_pipeline(RuntimeError(err_msg))

    if args.input2 and not args.paired_end:
        err_msg = "Incompatible settings: You specified single-end, but provided --input2."
        pm.fail_pipeline(RuntimeError(err_msg))

 
    ############################################################################
    #          Set up reference resources according to primary genome.         #
    ############################################################################
    if args.aligner.lower() == "bwa":
        GENOME_IDX_KEY = "bwa_index"
    else:
        GENOME_IDX_KEY = "bowtie2_index"
    check_list = [
        {"asset_name":"fasta", "seek_key":"chrom_sizes",
         "tag_name":"default", "arg":None, "user_arg":None,
         "required":True},
        {"asset_name":"fasta", "seek_key":None,
         "tag_name":"default", "arg":None, "user_arg":None,
         "required":True},
        {"asset_name":GENOME_IDX_KEY, "seek_key":None,
         "tag_name":"default", "arg":None, "user_arg":None,
         "required":True}
    ]
    # If user specifies TSS file, use that instead of the refgenie asset
    if not args.TSS_name:
        check_list.append(
            {"asset_name":"refgene_anno", "seek_key":"refgene_tss",
             "tag_name":"default", "arg":"TSS_name", "user_arg":"TSS-name",
             "required":False}
        )
    # If user specifies feature annotation file,
    # use that instead of the refgenie managed asset
    if not args.anno_name:
        check_list.append(
            {"asset_name":"feat_annotation", "seek_key":"feat_annotation",
            "tag_name":"default", "arg":"anno_name", "user_arg":"anno-name",
            "required":False}
        )
    # If user specifies blacklist file,
    # use that instead of the refgenie managed asset
    if not args.blacklist:
        check_list.append(
            {"asset_name":"blacklist", "seek_key":"blacklist",
            "tag_name":"default", "arg":"blacklist", "user_arg":"blacklist",
            "required":False}
        )
    res, rgc = _add_resources(args, res, check_list)

    # If the user specifies optional files, add those to our resources
    if (args.blacklist and os.path.isfile(args.blacklist) and
            os.stat(args.blacklist).st_size > 0):
        res.blacklist = args.blacklist
    if (args.frip_ref_peaks and os.path.isfile(args.frip_ref_peaks) and
            os.stat(args.frip_ref_peaks).st_size > 0):
        res.frip_ref_peaks = args.frip_ref_peaks
    if (args.TSS_name and os.path.isfile(args.TSS_name) and
            os.stat(args.TSS_name).st_size > 0):
        res.refgene_tss = args.TSS_name
    if (args.anno_name and os.path.isfile(args.anno_name) and
            os.stat(args.anno_name).st_size > 0):
        res.feat_annotation = args.anno_name

    # Adapter file can be set in the config; if left null, we use a default.
    res.adapters = res.adapters or tool_path("NexteraPE-PE.fa")

    # Report utilized assets
    assets_file = os.path.join(param.outfolder, "assets.tsv")
    for asset in res:
        message = "{}\t{}".format(asset, os.path.expandvars(res[asset]))
        report_message(pm, assets_file, message)
        
    # Report primary genome
    message = "genome\t{}".format(args.genome_assembly)
    report_message(pm, assets_file, message)


    ############################################################################
    #          Check that the input file(s) exist before continuing            #
    ############################################################################
    if os.path.isfile(args.input[0]) and os.stat(args.input[0]).st_size > 0:
        print("Local input file: " + args.input[0])
    elif os.path.isfile(args.input[0]) and os.stat(args.input[0]).st_size == 0:
        # The read1 file exists but is empty
        err_msg = "File exists but is empty: {}"
        pm.fail_pipeline(IOError(err_msg.format(args.input[0])))
    else:
        # The read1 file does not exist
        err_msg = "Could not find: {}"
        pm.fail_pipeline(IOError(err_msg.format(args.input[0])))

    if args.input2:
        if (os.path.isfile(args.input2[0]) and
                os.stat(args.input2[0]).st_size > 0):
            print("Local input file: " + args.input2[0])
        elif (os.path.isfile(args.input2[0]) and
                os.stat(args.input2[0]).st_size == 0):
            # The read1 file exists but is empty
            err_msg = "File exists but is empty: {}"
            pm.fail_pipeline(IOError(err_msg.format(args.input2[0])))
        else:
            # The read1 file does not exist
            err_msg = "Could not find: {}"
            pm.fail_pipeline(IOError(err_msg.format(args.input2[0])))

    container = None  # legacy


    ############################################################################
    #                      Grab and prepare input files                        #
    ############################################################################
    pm.report_result(
        "File_mb",
        round(ngstk.get_file_size(
              [x for x in [args.input, args.input2] if x is not None])), 2)
    pm.report_result("Read_type", args.single_or_paired)
    pm.report_result("Genome", args.genome_assembly)

    # ATACseq pipeline
    # Each (major) step should have its own subfolder
    raw_folder = os.path.join(param.outfolder, "raw")
    fastq_folder = os.path.join(param.outfolder, "fastq")

    pm.timestamp("### Merge/link and fastq conversion: ")
    # This command will merge multiple inputs so you can use multiple
    # sequencing lanes in a single pipeline run.
    local_input_files = ngstk.merge_or_link(
        [args.input, args.input2], raw_folder, args.sample_name)
    cmd, out_fastq_pre, unaligned_fastq = ngstk.input_to_fastq(
        local_input_files, args.sample_name, args.paired_end, fastq_folder,
        zipmode=True)
    print(cmd)
    pm.run(cmd, unaligned_fastq,
           follow=ngstk.check_fastq(
               local_input_files, unaligned_fastq, args.paired_end))
    pm.clean_add(out_fastq_pre + "*.fastq", conditional=True)

    if args.paired_end:
        untrimmed_fastq1 = unaligned_fastq[0]
        untrimmed_fastq2 = unaligned_fastq[1]
    else:
        untrimmed_fastq1 = unaligned_fastq
        untrimmed_fastq2 = None

    # Prepare alignment output folder
    map_genome_folder = os.path.join(param.outfolder,
                                     "aligned_" + args.genome_assembly)
    ngstk.make_dir(map_genome_folder)
    
    # Primary endpoint file following alignment and deduplication
    rmdup_bam = os.path.join(map_genome_folder,
                             args.sample_name + "_sort_dedup.bam")
    rmdup_idx = os.path.join(map_genome_folder,
                             args.sample_name + "_sort_dedup.bam.bai")


    ############################################################################
    #                          Begin adapter trimming                          #
    ############################################################################
    pm.timestamp("### Adapter trimming: ")

    # Create names for trimmed FASTQ files.
    if args.trimmer == "trimmomatic":
        trimming_prefix = os.path.join(fastq_folder, args.sample_name)
    else:
        trimming_prefix = out_fastq_pre
    trimmed_fastq = trimming_prefix + "_R1_trim.fastq"
    trimmed_fastq_R2 = trimming_prefix + "_R2_trim.fastq"
    fastqc_folder = os.path.join(param.outfolder, "fastqc")
    fastqc_report = os.path.join(fastqc_folder,
        trimming_prefix + "_R1_trim_fastqc.html")
    fastqc_report_R2 = os.path.join(fastqc_folder,
        trimming_prefix + "_R2_trim_fastqc.html")
    if ngstk.check_command(tools.fastqc):
        ngstk.make_dir(fastqc_folder)

    # Create trimming command(s).
    if args.trimmer == "pyadapt":
        if not args.paired_end:
            raise NotImplementedError(
                "pyadapt trimming requires paired-end reads.")
        # TODO: make pyadapt give options for output file name.
        trim_cmd_chunks = [
            tool_path("pyadapter_trim.py"),
            ("-a", untrimmed_fastq1),
            ("-b", untrimmed_fastq2),
            ("-o", out_fastq_pre),
            "-u"
        ]
        trim_cmd = build_command(trim_cmd_chunks)

    elif args.trimmer == "skewer":
        # Create the primary skewer command.
        # Don't compress output at this stage, because the pre-alignment mechanism
        # requires unzipped fastq.
        trim_cmd_chunks = [
            tools.skewer,  # + " --quiet"
            ("-f", "sanger"),
            ("-t", str(args.cores)),
            ("-m", "pe" if args.paired_end else "any"),
            ("-x", res.adapters),
            # "-z",  # compress output
            "--quiet",
            ("-o", out_fastq_pre),
            untrimmed_fastq1,
            untrimmed_fastq2 if args.paired_end else None
        ]
        trimming_command = build_command(trim_cmd_chunks)

        # Create the skewer file renaming commands.
        if args.paired_end:
            skewer_filename_pairs = \
                [("{}-trimmed-pair1.fastq".format(out_fastq_pre),
                 trimmed_fastq)]
            skewer_filename_pairs.append(
                ("{}-trimmed-pair2.fastq".format(out_fastq_pre),
                 trimmed_fastq_R2))
        else:
            skewer_filename_pairs = \
                [("{}-trimmed.fastq".format(out_fastq_pre), trimmed_fastq)]

        trimming_renaming_commands = [build_command(["mv", old, new])
                                      for old, new in skewer_filename_pairs]

        # Pypiper submits the commands serially.
        trim_cmd = [trimming_command] + trimming_renaming_commands

    else:
        # Default to trimmomatic.
        pm.info("trimmomatic local_input_files: {}".format(local_input_files))
        trim_cmd_chunks = [
            "{java} -Xmx{mem} -jar {trim} {PE} -threads {cores}".format(
                java=tools.java, mem=pm.mem,
                trim=tools.trimmomatic,
                PE="PE" if args.paired_end else "SE",
                cores=pm.cores),
            untrimmed_fastq1,
            untrimmed_fastq2 if args.paired_end else None,
            trimmed_fastq,
            trimming_prefix + "_R1_unpaired.fq" if args.paired_end else None,
            trimmed_fastq_R2 if args.paired_end else None,
            trimming_prefix + "_R2_unpaired.fq" if args.paired_end else None,
            "ILLUMINACLIP:" + res.adapters + ":2:30:10"
        ]
        trim_cmd = build_command(trim_cmd_chunks)

    def check_trim():
        pm.info("Evaluating read trimming")

        if args.paired_end and not trimmed_fastq_R2:
            pm.warning("Specified paired-end but no R2 file")

        n_trim = float(ngstk.count_reads(trimmed_fastq, args.paired_end))
        pm.report_result("Trimmed_reads", int(n_trim))
        try:
            rr = float(pm.get_stat("Raw_reads"))
        except:
            pm.warning("Can't calculate trim loss rate without raw read result.")
        else:
            pm.report_result(
                "Trim_loss_rate", round((rr - n_trim) * 100 / rr, 2))

        # Also run a fastqc (if installed/requested)
        if fastqc_folder and not args.skipqc:
            if fastqc_folder and os.path.isabs(fastqc_folder):
                ngstk.make_sure_path_exists(fastqc_folder)
            cmd = (tools.fastqc + " --noextract --outdir " +
                   fastqc_folder + " " + trimmed_fastq)
            pm.run(cmd, fastqc_report, nofail=False)
            pm.report_object("FastQC report r1", fastqc_report)

            if args.paired_end and trimmed_fastq_R2:
                cmd = (tools.fastqc + " --noextract --outdir " +
                       fastqc_folder + " " + trimmed_fastq_R2)
                pm.run(cmd, fastqc_report_R2, nofail=False)
                pm.report_object("FastQC report r2", fastqc_report_R2)

    if not os.path.exists(rmdup_bam) or args.new_start:
        pm.debug("trim_cmd: {}".format(trim_cmd))
        pm.run(trim_cmd, trimmed_fastq, follow=check_trim) 

    pm.clean_add(os.path.join(fastq_folder, "*.fastq"), conditional=True)
    pm.clean_add(os.path.join(fastq_folder, "*.log"), conditional=True)

    # Prepare variables for alignment step
    unmap_fq1 = trimmed_fastq
    unmap_fq2 = trimmed_fastq_R2


    ############################################################################
    #                    Map to any requested prealignments                    #
    ############################################################################

    # We recommend mapping to chrM (i.e. rCRSd) before primary genome alignment
    pm.timestamp("### Prealignments")
    # Keep track of the unmapped files in order to compress them after final
    # alignment.
    to_compress = []
    if len(args.prealignments) == 0:
        print("You may use `--prealignments` to align to references before "
              "the genome alignment step. See docs.")
    else:
        print("Prealignment assemblies: " + str(args.prealignments))
        # Loop through any prealignment references and map to them sequentially
        for reference in args.prealignments:
            genome_index = os.path.join(rgc.seek(reference, GENOME_IDX_KEY))
            if not os.path.exists(os.path.dirname(genome_index)):
                msg = "No {} index found in {}; skipping.".format(
                reference, os.path.dirname(genome_index))
                print(msg)
            else:            
                if not genome_index.endswith(reference):
                    genome_index = os.path.join(
                        os.path.dirname(rgc.seek(reference, GENOME_IDX_KEY)),
                        reference)
                    if args.aligner.lower() == "bwa":
                        genome_index += ".fa"
                if args.no_fifo:
                    unmap_fq1, unmap_fq2 = _align(
                        args, tools, args.paired_end, False,
                        unmap_fq1, unmap_fq2, reference,
                        assembly=genome_index,
                        outfolder=param.outfolder,
                        aligndir="prealignments",
                        bt2_opts_txt=param.bowtie2_pre.params,
                        bwa_opts_txt=param.bwa_pre.params)
                    to_compress.append(unmap_fq1)
                    if args.paired_end:
                        to_compress.append(unmap_fq2)
                else:
                    unmap_fq1, unmap_fq2 = _align(
                        args, tools, args.paired_end, True,
                        unmap_fq1, unmap_fq2, reference,
                        assembly=genome_index, 
                        outfolder=param.outfolder,
                        aligndir="prealignments",
                        bt2_opts_txt=param.bowtie2_pre.params,
                        bwa_opts_txt=param.bwa_pre.params)
                    to_compress.append(unmap_fq1)
                    if args.paired_end:
                        to_compress.append(unmap_fq2)

    pm.timestamp("### Compress all unmapped read files")
    for unmapped_fq in to_compress:
        # Compress unmapped fastq reads
        if not pypiper.is_gzipped_fastq(unmapped_fq) and not unmapped_fq == '':
            if os.path.exists(unmapped_fq):
                cmd = (ngstk.ziptool + " " + unmapped_fq)
                unmapped_fq = unmapped_fq + ".gz"
                pm.run(cmd, unmapped_fq)


    ############################################################################
    #                           Map to primary genome                          #
    ############################################################################
    pm.timestamp("### Map to genome")
    mapping_genome_bam = os.path.join(
        map_genome_folder, args.sample_name + "_sort.bam")
    mapping_genome_bam_temp = os.path.join(
        map_genome_folder, args.sample_name + "_temp.bam")
    failQC_genome_bam = os.path.join(
        map_genome_folder, args.sample_name + "_fail_qc.bam")
    unmap_genome_bam = os.path.join(
        map_genome_folder, args.sample_name + "_unmap.bam")

    if args.aligner.lower() == "bwa":
        if not param.bwa.params:
            bwa_options = " -M"
        else:
            bwa_options = param.bwa.params
    else:
        if not param.bowtie2.params:
            bt2_options = " --very-sensitive"
            if args.paired_end:
                bt2_options += " -X 2000"
        else:
            bt2_options = param.bowtie2.params    

    # samtools sort needs a temporary directory
    tempdir = tempfile.mkdtemp(dir=map_genome_folder)
    os.chmod(tempdir, 0o771)
    pm.clean_add(tempdir)

    # If there are no prealignments, unmap_fq1 will be unzipped
    if os.path.exists(unmap_fq1 + ".gz"):
        unmap_fq1 = unmap_fq1 + ".gz"
    if os.path.exists(unmap_fq2 + ".gz"):
        unmap_fq2 = unmap_fq2 + ".gz"

    genome_index = os.path.join(rgc.seek(args.genome_assembly, GENOME_IDX_KEY))          
    if not genome_index.endswith(args.genome_assembly):
        genome_index = os.path.join(
            os.path.dirname(rgc.seek(args.genome_assembly, GENOME_IDX_KEY)),
            args.genome_assembly)
        if args.aligner.lower() == "bwa":
            genome_index += ".fa"

    if args.aligner.lower() == "bwa":
        cmd = tools.bwa + " mem -t " + str(pm.cores)
        cmd += " " + bwa_options
        cmd += " " + genome_index
        cmd += " " + unmap_fq1
        if args.paired_end:
            cmd += " " + unmap_fq2
        cmd += " | " + tools.samtools + " view -bS - -@ 1"  # convert to bam
        cmd += " | " + tools.samtools + " sort - -@ 1"      # sort output
        cmd += " -T " + tempdir
        cmd += " -o " + mapping_genome_bam_temp
    else:
        cmd = tools.bowtie2 + " -p " + str(pm.cores)
        cmd += " " + bt2_options
        cmd += " --rg-id " + args.sample_name
        cmd += " -x " + genome_index
        if args.paired_end:
            cmd += " -1 " + unmap_fq1 + " -2 " + unmap_fq2
        else:
            cmd += " -U " + unmap_fq1
        cmd += " | " + tools.samtools + " view -bS - -@ 1 "
        cmd += " | " + tools.samtools + " sort - -@ 1"
        cmd += " -T " + tempdir
        cmd += " -o " + mapping_genome_bam_temp

    # Split genome mapping result bamfile into two: high-quality aligned
    # reads (keepers) and unmapped reads (in case we want to analyze the
    # altogether unmapped reads)
    # Default (samtools.params): skip alignments with MAPQ less than 10 (-q 10)
    cmd2 = (tools.samtools + " view -b " + param.samtools.params + " -@ " +
            str(pm.cores) + " -U " + failQC_genome_bam + " ")
    if args.paired_end:
        # add a step to accept only reads mapped in proper pair
        cmd2 += "-f 2 "

    cmd2 += mapping_genome_bam_temp + " > " + mapping_genome_bam

    def check_alignment_genome():
        mr = ngstk.count_mapped_reads(mapping_genome_bam_temp, args.paired_end)
        ar = ngstk.count_mapped_reads(mapping_genome_bam, args.paired_end)
        rr = float(pm.get_stat("Raw_reads"))
        tr = float(pm.get_stat("Trimmed_reads"))
        pm.report_result("Mapped_reads", mr)
        pm.report_result("QC_filtered_reads",
                         round(float(mr)) - round(float(ar)))
        pm.report_result("Aligned_reads", ar)
        pm.report_result("Alignment_rate", round(float(ar) * 100 /
                         float(tr), 2))
        pm.report_result("Total_efficiency", round(float(ar) * 100 /
                         float(rr), 2))

    pm.run([cmd, cmd2], rmdup_bam, follow=check_alignment_genome)

    # Index the temporary bam file and the sorted bam file
    temp_mapping_index   = os.path.join(mapping_genome_bam_temp + ".bai")
    mapping_genome_index = os.path.join(mapping_genome_bam + ".bai")
    cmd1 = tools.samtools + " index " + mapping_genome_bam_temp
    cmd2 = tools.samtools + " index " + mapping_genome_bam
    pm.run([cmd1, cmd2], rmdup_idx)
    pm.clean_add(temp_mapping_index)

    # If first run, use the temp bam file
    if (os.path.isfile(mapping_genome_bam_temp) and
            os.stat(mapping_genome_bam_temp).st_size > 0):
        bam_file = mapping_genome_bam_temp
    # Otherwise, use the final bam file previously generated
    else:
        bam_file = mapping_genome_bam

    # Determine mitochondrial read counts
    mito_name = ["chrM", "ChrM", "ChrMT", "chrMT", "M", "MT", "rCRSd"]

    if not pm.get_stat("Mitochondrial_reads") or args.new_start:
        cmd = (tools.samtools + " idxstats " + bam_file + " | grep")
        for name in mito_name:
            cmd += " -we '" + name + "'"
        cmd += "| cut -f 3"
        mr = pm.checkprint(cmd)

        # If there are mitochondrial reads, report and remove them
        if mr and float(mr.strip()) != 0:
            pm.report_result("Mitochondrial_reads", round(float(mr)))
            noMT_mapping_genome_bam = os.path.join(
                map_genome_folder, args.sample_name + "_noMT.bam")
            chr_bed = os.path.join(map_genome_folder, "chr_sizes.bed")

            cmd1 = (tools.samtools + " idxstats " + mapping_genome_bam +
                    " | cut -f 1-2 | awk '{print $1, 0, $2}' | grep")
            for name in mito_name:
                cmd1 += " -vwe '" + name + "'"
            cmd1 += (" > " + chr_bed)
            cmd2 = (tools.samtools + " view -L " + chr_bed + " -b -@ " +
                    str(pm.cores) + " " + mapping_genome_bam + " > " +
                    noMT_mapping_genome_bam)
            cmd3 = ("mv " + noMT_mapping_genome_bam + " " + mapping_genome_bam)
            # Reindex the sorted bam file now that mito reads are removed
            cmd4 = tools.samtools + " index " + mapping_genome_bam

            pm.run([cmd1, cmd2, cmd3, cmd4], noMT_mapping_genome_bam)
            pm.clean_add(chr_bed)
        else:
            pm.report_result("Mitochondrial_reads", 0)


    ############################################################################
    #         Calculate quality control metrics for the alignment file         #
    ############################################################################
    pm.timestamp("### Calculate NRF, PBC1, and PBC2")
    QC_folder = os.path.join(param.outfolder, "QC_" + args.genome_assembly)
    ngstk.make_dir(QC_folder)

    bamQC = os.path.join(QC_folder, args.sample_name + "_bamQC.tsv")
    cmd = tool_path("bamQC.py")
    cmd += " -i " + mapping_genome_bam
    cmd += " -c " + str(pm.cores)
    cmd += " -o " + bamQC

    def report_bam_qc(bamqc_log):
        # Reported BAM QC metrics via the bamQC metrics file
        if os.path.isfile(bamqc_log):
            cmd1 = ("awk '{ for (i=1; i<=NF; ++i) {" +
                    " if ($i ~ \"NRF\") c=i } getline; print $c }' " +
                    bamqc_log)
            cmd2 = ("awk '{ for (i=1; i<=NF; ++i) {" +
                    " if ($i ~ \"PBC1\") c=i } getline; print $c }' " +
                    bamqc_log)
            cmd3 = ("awk '{ for (i=1; i<=NF; ++i) {" +
                    " if ($i ~ \"PBC2\") c=i } getline; print $c }' " +
                    bamqc_log)
            nrf = pm.checkprint(cmd1)
            pbc1 = pm.checkprint(cmd2)
            pbc2 = pm.checkprint(cmd3)
        else:
            # there were no successful chromosomes yielding results
            nrf = 0
            pbc1 = 0
            pbc2 = 0

        pm.report_result("NRF", round(float(nrf),2))
        pm.report_result("PBC1", round(float(pbc1),2))
        pm.report_result("PBC2", round(float(pbc2), 2))

    pm.run(cmd, bamQC, follow=lambda: report_bam_qc(bamQC))

    # Now produce the unmapped file
    def count_unmapped_reads():
        # Report total number of unmapped reads (-f 4)
        cmd = (tools.samtools + " view -c -f 4 -@ " + str(pm.cores) +
               " " + mapping_genome_bam_temp)
        ur = pm.checkprint(cmd)
        pm.report_result("Unmapped_reads", round(float(ur)))

    unmap_cmd = tools.samtools + " view -b -@ " + str(pm.cores)
    if args.paired_end:
        # require both read and mate unmapped
        unmap_cmd += " -f 12 "
    else:
        # require only read unmapped
        unmap_cmd += " -f 4 "

    unmap_cmd += " " + mapping_genome_bam_temp + " > " + unmap_genome_bam
    
    if not pm.get_stat("Unmapped_reads") or args.new_start:
        pm.run(unmap_cmd, unmap_genome_bam, follow=count_unmapped_reads)

    # Remove temporary bam file from unmapped file production
    pm.clean_add(mapping_genome_bam_temp)

    pm.timestamp("### Remove duplicates and produce signal tracks")

    def estimate_lib_size(dedup_log):
        # In millions of reads; contributed by Ryan
        # NOTE: from Picard manual: without optical duplicate counts,
        #       library size estimation will be inaccurate.
        cmd = ("awk -F'\t' -f " + tool_path("extract_picard_lib.awk") +
               " " + dedup_log)
        picard_est_lib_size = pm.checkprint(cmd)
        pm.report_result("Picard_est_lib_size", picard_est_lib_size)

    def post_dup_aligned_reads(dedup_log):
        if args.deduplicator == "picard":
            cmd = ("grep -A2 'METRICS CLASS' " + dedup_log +
                   " | tail -n 1 | awk '{print $(NF-3)}'")
        elif args.deduplicator == "samblaster":
            cmd = ("grep 'Removed' " + dedup_log +
                   " | tr -s ' ' | cut -f 3 -d ' '")
        elif args.deduplicator == "samtools":
            cmd = ("grep 'DUPLICATE TOTAL' " + dedup_log +
                   " | tr -s ' ' | cut -f 3 -d ' '")
        else:
            cmd = ("grep 'Removed' " + dedup_log +
                   " | tr -s ' ' | cut -f 3 -d ' '")

        dr = pm.checkprint(cmd)
        ar = float(pm.get_stat("Aligned_reads"))
        rr = float(pm.get_stat("Raw_reads"))
        tr = float(pm.get_stat("Trimmed_reads"))

        if not dr and not dr.strip():
            pm.info("DEBUG: dr didn't work correctly")
            dr = ar
        if args.deduplicator == "samtools":
            dr = float(dr)/2
        
        pdar = float(ar) - float(dr)
        dar = round(float(pdar) * 100 / float(tr), 2)
        dte = round(float(pdar) * 100 / float(rr), 2)
        
        pm.report_result("Duplicate_reads", dr)
        pm.report_result("Dedup_aligned_reads", pdar)
        pm.report_result("Dedup_alignment_rate", dar)
        pm.report_result("Dedup_total_efficiency", dte)

    metrics_file = os.path.join(
        map_genome_folder, args.sample_name + "_dedup_metrics_bam.txt")
    dedup_log = os.path.join(
        map_genome_folder, args.sample_name + "_dedup_metrics_log.txt")

    # samtools sort needs a temporary directory
    tempdir = tempfile.mkdtemp(dir=map_genome_folder)
    os.chmod(tempdir, 0o771)
    pm.clean_add(tempdir)

    if args.deduplicator == "picard":
        cmd1 = (tools.java + " -Xmx" + str(pm.javamem) + " -jar " + 
                tools.picard + " MarkDuplicates")
        cmd1 += " INPUT=" + mapping_genome_bam
        cmd1 += " OUTPUT=" + rmdup_bam
        cmd1 += " METRICS_FILE=" + metrics_file
        cmd1 += " VALIDATION_STRINGENCY=LENIENT"
        cmd1 += " ASSUME_SORTED=true REMOVE_DUPLICATES=true > " + dedup_log
        cmd2 = tools.samtools + " index " + rmdup_bam
    elif args.deduplicator == "samblaster":
        nProc = max(int(pm.cores / 4), 1)
        samblaster_cmd_chunks = [
            "{} sort -n -@ {}".format(tools.samtools, str(nProc)),
            ("-T", tempdir),
            mapping_genome_bam,
            "|",
            "{} view -h - -@ {}".format(tools.samtools, str(nProc)),
            "|",
            "{} -r --ignoreUnmated 2> {}".format(tools.samblaster, dedup_log),
            "|",
            "{} view -b - -@ {}".format(tools.samtools, str(nProc)),
            "|",
            "{} sort - -@ {}".format(tools.samtools, str(nProc)),
            ("-T", tempdir),
            ("-o", rmdup_bam)
        ]
        cmd1 = build_command(samblaster_cmd_chunks)
        cmd2 = tools.samtools + " index " + rmdup_bam
        # no separate metrics file with samblaster
        metrics_file = dedup_log
    elif args.deduplicator == "samtools":
        nProc = max(int(pm.cores / 4), 1)
        samtools_cmd_chunks = [
            "{} sort -n -@ {} -T {}".format(tools.samtools, str(nProc),
                                            tempdir),
            mapping_genome_bam,
            "|",
            "{} fixmate -@ {} -m - - ".format(tools.samtools, str(nProc)),
            "|",
            "{} sort -@ {} -T {}".format(tools.samtools, str(nProc), tempdir),
            "|",
            "{} markdup -@ {} -T {} -rs -f {} - - ".format(tools.samtools,
                                                           str(nProc),
                                                           tempdir,
                                                           dedup_log),
            "|",
            "{} view -b - -@ {}".format(tools.samtools, str(nProc)),
            "|",
            "{} sort - -@ {} -T {}".format(tools.samtools, str(nProc), tempdir),
            ("-o", rmdup_bam)
        ]
        cmd1 = build_command(samtools_cmd_chunks)
        cmd2 = tools.samtools + " index " + rmdup_bam
        metrics_file = dedup_log
    else:
        pm.info("PEPATAC could not determine a valid deduplicator tool")
        pm.stop_pipeline()

    pm.run([cmd1, cmd2], rmdup_bam,
           follow=lambda: post_dup_aligned_reads(metrics_file))
    
    
    ############################################################################
    #           Determine distribution of reads across nucleosomes             #
    ############################################################################
    pm.timestamp("### Calculate distribution of reads across nucleosomes")

    # Use cutoff method original proposed in Buenrostro et al. 2013
    NFR_bam = os.path.join(map_genome_folder, args.sample_name + "_NFR.bam")
    mono_bam = os.path.join(map_genome_folder, args.sample_name + "_mono.bam")
    di_bam = os.path.join(map_genome_folder, args.sample_name + "_di.bam")
    tri_bam = os.path.join(map_genome_folder, args.sample_name + "_tri.bam")
    poly_bam = os.path.join(map_genome_folder, args.sample_name + "_poly.bam")
    
    # Need parenthesis outside of substr for pypiper split_shell_cmd parsing
    cmd1 = (tools.samtools + " view -h " + rmdup_bam + " | awk " +
            "'(substr($0,1,1)==\"@\" || ($9>= -100 && $9<=100))' | " +
            tools.samtools + " view -b > " + NFR_bam)
    cmd2 = (tools.samtools + " view -h " + rmdup_bam + " | awk " +
            "'(substr($0,1,1)==\"@\" || ($9>= 180 && $9<=247) || " +
            "($9<=-180 && $9>=-247))' | " + tools.samtools + " view -b > " +
            mono_bam)
    cmd3 = (tools.samtools + " view -h " + rmdup_bam + " | awk " +
            "'(substr($0,1,1)==\"@\" || ($9>= 315 && $9<=473) || " +
            "($9<=-315 && $9>=-473))' | " + tools.samtools + " view -b > " +
            di_bam)
    cmd4 = (tools.samtools + " view -h " + rmdup_bam + " | awk " +
            "'(substr($0,1,1)==\"@\" || ($9>= 558 && $9<=615) || " +
            "($9<=-558 && $9>=-615))' | " + tools.samtools + " view -b > " +
            tri_bam)        
    cmd5 = (tools.samtools + " view -h " + rmdup_bam + " | awk " +
            "'(substr($0,1,1)==\"@\" || ($9>= 615 || $9<=-615))' | " +
            tools.samtools + " view -b > " + poly_bam)
    pm.run(cmd1, NFR_bam)
    pm.run(cmd2, mono_bam)
    pm.run(cmd3, di_bam)
    pm.run(cmd4, tri_bam)
    pm.run(cmd5, poly_bam)
    
    cmd1 = tools.samtools + " view -c " + NFR_bam
    cmd2 = tools.samtools + " view -c " + mono_bam
    cmd3 = tools.samtools + " view -c " + di_bam
    cmd4 = tools.samtools + " view -c " + tri_bam
    cmd5 = tools.samtools + " view -c " + poly_bam
    nfr = pm.checkprint(cmd1)
    mono = pm.checkprint(cmd2)
    di = pm.checkprint(cmd3)
    tri = pm.checkprint(cmd4)
    poly = pm.checkprint(cmd5)
    
    dar = float(pm.get_stat("Dedup_aligned_reads"))
    NFR_frac = round(float(nfr) / float(dar), 2)
    mono_frac = round(float(mono) / float(dar), 2)
    di_frac = round(float(di) / float(dar), 2)
    tri_frac = round(float(tri) / float(dar), 2)
    poly_frac = round(float(poly) / float(dar), 2)
    
    pm.report_result("NFR_frac", NFR_frac)
    pm.report_result("mono_frac", mono_frac)
    pm.report_result("di_frac", di_frac)
    pm.report_result("tri_frac", tri_frac)
    pm.report_result("poly_frac", poly_frac)

    ############################################################################
    #       Determine maximum read length and add seqOutBias resource          #
    ############################################################################
    if not pm.get_stat("Read_length") or args.new_start:
        if (os.path.isfile(mapping_genome_bam)
            and os.stat(mapping_genome_bam).st_size > 0):
            cmd = (tools.samtools + " stats " + mapping_genome_bam +
                   " | grep '^SN' | cut -f 2- | grep 'maximum length:' " +
                   "| cut -f 2-")
            read_len = int(pm.checkprint(cmd))
        else:
            pm.warning("{} could not be found.".format(mapping_genome_bam))
            pm.stop_pipeline()
        pm.report_result("Read_length", read_len)
    else:
        read_len = int(pm.get_stat("Read_length"))

    # At this point we can check for seqOutBias required indicies.
    # Can't do it earlier because we haven't determined the read_length of 
    # interest for mappability purposes.
    if args.sob:
        pm.debug("read_len: {}".format(read_len))  # DEBUG
        search_asset = [{"asset_name":"tallymer_index",
                         "seek_key":"search_file",
                         "tag_name":read_len,
                         "arg":"search_file",
                         "user_arg":"search-file",
                         "required":True}]
        res, rgc = _add_resources(args, res, search_asset)

    # Calculate size of genome
    if not pm.get_stat("Genome_size") or args.new_start:
        genome_size = int(pm.checkprint(
            ("awk '{sum+=$2} END {printf \"%.0f\", sum}' " +
             res.chrom_sizes)))
        pm.report_result("Genome_size", genome_size)
    else:
        genome_size = int(pm.get_stat("Genome_size"))


    ############################################################################
    #                     Calculate library complexity                         #
    ############################################################################
    preseq_output = os.path.join(
        QC_folder, args.sample_name + "_preseq_out.txt")
    preseq_yield = os.path.join(
        QC_folder, args.sample_name + "_preseq_yield.txt")
    preseq_counts = os.path.join(
        QC_folder, args.sample_name + "_preseq_counts.txt")
    preseq_pdf = os.path.join(
        QC_folder, args.sample_name + "_preseq_plot.pdf")
    preseq_plot = os.path.join(
        QC_folder, args.sample_name + "_preseq_plot")
    preseq_png = os.path.join(
        QC_folder, args.sample_name + "_preseq_plot.png")

    if not os.path.exists(preseq_pdf) or args.new_start:
        if not os.path.exists(mapping_genome_index):
            cmd = tools.samtools + " index " + mapping_genome_bam
            pm.run(cmd, mapping_genome_index)
            pm.clean_add(mapping_genome_index)

        pm.timestamp("### Calculate library complexity")

        cmd1 = (tools.preseq + " c_curve -v -o " + preseq_output +
                " -B " + mapping_genome_bam)
        pm.run(cmd1, preseq_output)

        cmd2 = (tools.preseq + " lc_extrap -v -o " + preseq_yield +
                " -B " + mapping_genome_bam)
        pm.run(cmd2, preseq_yield, nofail=True)

        if os.path.exists(preseq_yield):
            cmd3 = ("echo '" + preseq_yield +
                    " '$(" + tools.samtools + " view -c -F 4 " + 
                    mapping_genome_bam + ")" + "' '" +
                    "$(" + tools.samtools + " view -c -F 4 " +
                    rmdup_bam + ") > " + preseq_counts)

            pm.run(cmd3, preseq_counts)

            cmd = (tools.Rscript + " " + tool_path("PEPATAC.R") +
                   " preseq " + "-i " + preseq_yield)
            cmd += (" -r " + preseq_counts + " -o " + preseq_plot)

            pm.run(cmd, [preseq_pdf, preseq_png], nofail=True)

            pm.report_object("Library complexity", preseq_pdf,
                             anchor_image=preseq_png)

            if not pm.get_stat('Frac_exp_unique_at_10M') or args.new_start:
                # Report the expected unique at 10M reads
                cmd = ("grep -w '^10000000' " + preseq_yield +
                       " | awk '{print $2}'")
                expected_unique = pm.checkprint(cmd)
                if expected_unique:
                    fraction_unique = float(expected_unique)/float(10000000)
                    pm.report_result("Frac_exp_unique_at_10M",
                                     round(fraction_unique, 4))
        else:
            print("Unable to calculate library complexity.")

    ############################################################################
    #                          Determine TSS enrichment                        #
    ############################################################################
    if not os.path.exists(res.refgene_tss):
        print("Skipping TSS -- TSS enrichment requires TSS annotation file: {}"
              .format(res.refgene_tss))
    else:
        pm.timestamp("### Calculate TSS enrichment")

        Tss_enrich = os.path.join(QC_folder, args.sample_name +
                                  "_TSS_enrichment.txt")
        cmd = tool_path("pyTssEnrichment.py")
        cmd += " -a " + rmdup_bam + " -b " + res.refgene_tss + " -p ends"
        cmd += " -c " + str(pm.cores)
        cmd += " -z -v -s 6 -o " + Tss_enrich
        pm.run(cmd, Tss_enrich, nofail=True)

        if not pm.get_stat('TSS_score') or args.new_start:
            with open(Tss_enrich) as f:
                floats = list(map(float, f))
            try:
                # If the TSS enrichment is 0, don't report
                list_len = 0.05*float(len(floats))
                normTSS = [x / (sum(floats[1:int(list_len)]) /
                           len(floats[1:int(list_len)])) for x in floats]
                max_index = normTSS.index(max(normTSS))

                if (((normTSS[max_index]/normTSS[max_index-1]) > 1.5) and
                    ((normTSS[max_index]/normTSS[max_index+1]) > 1.5)):
                    tmpTSS = list(normTSS)
                    del tmpTSS[max_index]
                    max_index = tmpTSS.index(max(tmpTSS)) + 1

                Tss_score = round(
                    (sum(normTSS[int(max_index-50):int(max_index+50)])) /
                    (len(normTSS[int(max_index-50):int(max_index+50)])), 1)

                pm.report_result("TSS_score", round(Tss_score, 1))
            except ZeroDivisionError:
                pm.report_result("TSS_score", 0)
                pass
        
        # Call Rscript to plot TSS Enrichment
        Tss_pdf = os.path.join(QC_folder,  args.sample_name +
                               "_TSS_enrichment.pdf")
        Tss_png = os.path.join(QC_folder,  args.sample_name +
                               "_TSS_enrichment.png")
        cmd = (tools.Rscript + " " + tool_path("PEPATAC.R") + 
               " tss -i " + Tss_enrich)
        pm.run(cmd, Tss_pdf, nofail=True)

        pm.report_object("TSS enrichment", Tss_pdf, anchor_image=Tss_png)


    ############################################################################
    #                         Fragment distribution                            #
    ############################################################################
    if args.paired_end:
        pm.timestamp("### Plot fragment distribution")
        frag_len = os.path.join(QC_folder,
                                args.sample_name + "_fragLen.txt")
        cmd1 = build_command([tools.perl,
                              tool_path("fragment_length_dist.pl"),
                              rmdup_bam,
                              frag_len])

        fragL_count = os.path.join(QC_folder,
                                   args.sample_name + "_fragCount.txt")
        cmd2 = ("sort -n  " + frag_len + " | uniq -c  > " + fragL_count)

        fragL_dis1 = os.path.join(QC_folder, args.sample_name +
                                  "_fragLenDistribution.pdf")
        fragL_png = os.path.join(QC_folder, args.sample_name +
                                 "_fragLenDistribution.png")
        fragL_dis2 = os.path.join(QC_folder, args.sample_name +
                                  "_fragLenDistribution.txt")

        cmd3 = (tools.Rscript + " " + tool_path("PEPATAC.R") +
                " frag -l " + frag_len + " -c " + fragL_count +
                " -p " + fragL_dis1 + " -t " + fragL_dis2)

        pm.run([cmd1, cmd2, cmd3], fragL_dis1, nofail=True)
        pm.report_object("Fragment distribution", fragL_dis1,
                         anchor_image=fragL_png)
    else: 
        print("Fragment distribution requires paired-end data")


    ############################################################################
    #                        Extract genomic features                          #
    ############################################################################
    # Generate local unzipped annotation file
    anno_local = os.path.join(raw_folder,
                              args.genome_assembly + "_annotations.bed")
    anno_zip = os.path.join(raw_folder,
                            args.genome_assembly + "_annotations.bed.gz")

    if (not os.path.exists(anno_local) and
        not os.path.exists(anno_zip) and
        os.path.exists(res.feat_annotation) or
        args.new_start):

        if res.feat_annotation.endswith(".gz"):
            cmd1 = ("ln -sf " + res.feat_annotation + " " + anno_zip)
            cmd2 = (ngstk.ziptool + " -d -c " + anno_zip +
                    " > " + anno_local)
            pm.run([cmd1, cmd2], anno_local)
            pm.clean_add(anno_local)
        elif res.feat_annotation.endswith(".bed"):
            cmd = ("ln -sf " + res.feat_annotation + " " + anno_local)
            pm.run(cmd, anno_local)
            pm.clean_add(anno_local)
        else:
            print("Skipping read and peak annotation...")
            print("This requires a {} annotation file."
                  .format(args.genome_assembly))
            print("Could not find the feat_annotation asset {}.`"
                  .format(str(os.path.dirname(res.feat_annotation))))


    ############################################################################
    #            Remove all but final output files to save space               #
    ############################################################################
    if args.lite:
        # Remove everything but ultimate outputs
        pm.clean_add(frag_len)
        pm.clean_add(fragL_dis2)
        pm.clean_add(fragL_count)
        pm.clean_add(Tss_enrich)
        pm.clean_add(mapping_genome_bam)
        pm.clean_add(mapping_genome_index)
        pm.clean_add(failQC_genome_bam)
        pm.clean_add(unmap_genome_bam)
        pm.clean_add(NFR_bam)
        pm.clean_add(mono_bam)
        pm.clean_add(di_bam)
        pm.clean_add(tri_bam)
        for unmapped_fq in to_compress:
            if not unmapped_fq:
                pm.clean_add(unmapped_fq + ".gz")


    ############################################################################
    #                            PIPELINE COMPLETE!                            #
    ############################################################################
    pm.stop_pipeline()


if __name__ == '__main__':
    pm = None
    # TODO: remove once ngstk become less instance-y, more function-y.
    ngstk = None
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit("Pipeline aborted")
