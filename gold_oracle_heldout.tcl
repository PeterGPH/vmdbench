# gold_oracle_heldout.tcl — gold for tasks OUTSIDE the semantic-tool coverage.
# None of these is a vmd_measure metric, so the agent must write raw Tcl. This tests
# whether the #1-#3 improvements generalize beyond tool-covered operations.
#
#   GOLD_STRUCT=/path/structure vmd -dispdev text -e gold_oracle_heldout.tcl

proc emit {k v} { puts "GOLD $k $v" }
proc idx1 {seltext} { set s [atomselect top $seltext]; set i [lindex [$s get index] 0]; $s delete; return $i }

set structfile ""
if {[info exists env(GOLD_STRUCT)]} { set structfile $env(GOLD_STRUCT) }
if {$structfile eq "" || ![file exists $structfile]} { puts "GOLD_ERROR structfile-not-found '$structfile'"; quit }
set mol [mol new $structfile waitfor all]

# hydrogen bonds within the protein (cutoff 3.0 A, angle 20 deg) — count = len of first list
if {[catch { set p [atomselect top protein]; set hb [measure hbonds 3.0 20 $p]; emit hbonds [llength [lindex $hb 0]]; $p delete } e]} { puts "GOLD_ERROR hbonds $e" }

# angle (deg) formed by CA atoms of residues 1, 2, 3
if {[catch { emit angle123 [format %.2f [measure angle [list [idx1 "name CA and resid 1"] [idx1 "name CA and resid 2"] [idx1 "name CA and resid 3"]]]] } e]} { puts "GOLD_ERROR angle123 $e" }

# phi backbone dihedral (deg) of residue 5: C(4)-N(5)-CA(5)-C(5)
if {[catch { emit phi5 [format %.2f [measure dihed [list [idx1 "name C and resid 4"] [idx1 "name N and resid 5"] [idx1 "name CA and resid 5"] [idx1 "name C and resid 5"]]]] } e]} { puts "GOLD_ERROR phi5 $e" }

# x-coordinate (A) of the protein center of mass
if {[catch { set p [atomselect top protein]; emit com_x [format %.4f [lindex [measure center $p weight mass] 0]]; $p delete } e]} { puts "GOLD_ERROR com_x $e" }

# largest bounding-box dimension (A) of the protein
if {[catch {
    set p [atomselect top protein]; set mm [measure minmax $p]; $p delete
    set a [lindex $mm 0]; set b [lindex $mm 1]
    set dx [expr {[lindex $b 0]-[lindex $a 0]}]; set dy [expr {[lindex $b 1]-[lindex $a 1]}]; set dz [expr {[lindex $b 2]-[lindex $a 2]}]
    emit max_extent [format %.4f [expr {max($dx,max($dy,$dz))}]]
} e]} { puts "GOLD_ERROR max_extent $e" }

# number of glycine (GLY) residues
if {[catch { set p [atomselect top "protein and resname GLY"]; emit n_gly [llength [lsort -unique [$p get residue]]]; $p delete } e]} { puts "GOLD_ERROR n_gly $e" }

quit
