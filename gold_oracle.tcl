# gold_oracle.tcl — compute reference ("gold") analysis values for 1CRN, headless.
#
#   GOLD_STRUCT=/path/to/data/1CRN.cif \
#     /Applications/VMD.app/Contents/vmd/vmd_MACOSXARM64 -dispdev text -e gold_oracle.tcl
#
# Emits parseable lines:  GOLD <key> <value>   (everything else is VMD noise).
# Each measurement is wrapped in catch{} so one failure can't sink the rest.

proc emit {k v} { puts "GOLD $k $v" }

proc idx1 {mol seltext} {
    set s [atomselect $mol $seltext]
    set i [lindex [$s get index] 0]
    $s delete
    return $i
}

set structfile ""
if {[info exists env(GOLD_STRUCT)]} { set structfile $env(GOLD_STRUCT) }
if {$structfile eq "" || ![file exists $structfile]} {
    puts "GOLD_ERROR structfile-not-found '$structfile'"
    quit
}

set mol [mol new $structfile waitfor all]

# case_8 — radius of gyration of the protein (gold ~9.67 Å)
if {[catch {
    set sel [atomselect $mol "protein"]
    emit rgyr [format %.4f [measure rgyr $sel]]
    $sel delete
} err]} { puts "GOLD_ERROR rgyr $err" }

# case_7 — RMSD of the structure vs itself = 0 (single static frame)
if {[catch {
    set a [atomselect $mol "all"]
    emit rmsd_self [format %.4f [measure rmsd $a $a]]
    $a delete
} err]} { puts "GOLD_ERROR rmsd_self $err" }

# case_9 — CA(resid 1) to CA(resid 10) distance
if {[catch {
    set i [idx1 $mol "name CA and resid 1"]
    set j [idx1 $mol "name CA and resid 10"]
    emit dist_ca1_ca10 [format %.4f [measure bond [list $i $j]]]
} err]} { puts "GOLD_ERROR dist_ca1_ca10 $err" }

# case_9 — phi/psi backbone dihedrals of residue 5
if {[catch {
    set Cprev [idx1 $mol "name C and resid 4"]
    set N     [idx1 $mol "name N and resid 5"]
    set CA    [idx1 $mol "name CA and resid 5"]
    set C     [idx1 $mol "name C and resid 5"]
    set Nnext [idx1 $mol "name N and resid 6"]
    emit phi_resid5 [format %.2f [measure dihed [list $Cprev $N $CA $C]]]
    emit psi_resid5 [format %.2f [measure dihed [list $N $CA $C $Nnext]]]
} err]} { puts "GOLD_ERROR phipsi $err" }

# case_10 — atom pairs within 8 Å among the protein (contacts; definition-sensitive)
if {[catch {
    set sel [atomselect $mol "protein"]
    set res [measure contacts 8.0 $sel]
    emit contacts_8 [llength [lindex $res 0]]
    $sel delete
} err]} { puts "GOLD_ERROR contacts_8 $err" }

quit
