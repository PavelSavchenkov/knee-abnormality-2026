# Official label definitions

Source: "Knee Abnormality Detection AI Challenge Overview", competition host post (Po-Hao "Howard" Chen, courtesy of Dr. Jacob Kazam),
https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733343 . Text below is copied verbatim.

## Anatomy used by the labels

For the purposes of this challenge, the knee is considered in three compartments. The medial compartment lies between the medial femoral condyle and the medial tibial plateau, on the inner side of the knee. The lateral compartment lies between the lateral femoral condyle and the lateral tibial plateau, on the outer side. The patellofemoral compartment lies between the patella and the femoral trochlea, the groove on the front of the femur. Osteoarthritis, or cartilage loss, is assessed separately in each of these three compartments.

Four ligaments stabilize the knee. The anterior cruciate ligament (ACL) and posterior cruciate ligament (PCL) lie within the joint, in the intercondylar notch, and control front-to-back stability and rotation. The medial collateral ligament (MCL) and lateral collateral ligament (LCL) run along the inner and outer sides of the knee and resist side-to-side stress. This challenge focuses on tears of the ACL and the MCL.

Between the femur and tibia sit two menisci, the medial meniscus and the lateral meniscus, C-shaped wedges of fibrocartilage that cushion the joint, absorb shock, and improve the fit between the rounded femur and the relatively flat tibia. Tears of the medial and lateral meniscus are evaluated separately.

Other relevant structures include the bone marrow within the femur, tibia, and patella, where a bruise from impact is called a bone contusion (or bone marrow edema) and a break in the bone is a fracture; and the soft tissues behind the knee, where a fluid-filled outpouching of the joint lining is called a Baker, or popliteal, cyst.

For the purpose of the 2026 Knee Abnormality Challenge, a “fluid sensitive” sequence refers to one in which edema, hemorrhage, and other types of fluid appear bright and fat is suppressed in some way.

## Label description

Models are evaluated on twelve binary labels, each indicating the presence or absence of a specific finding in the imaged knee. The labels, and the criteria used by the annotating radiologists, are summarized below. In each case, ambiguous or borderline findings (“on the fence”) were graded as negative to favor specificity.

ACL tear: A high-grade partial or full-thickness tear of the anterior cruciate ligament, meaning complete discontinuity of the ligament, or more than 50 percent of fibers disrupted, with or without secondary signs such as characteristic pivot-shift bone contusions. Mild signal change, degeneration, or thickening without discontinuity is graded negative.

MCL tear: A high-grade partial or complete acute tear of the medial collateral ligament, with disrupted fibers and edema within and adjacent to the ligament. Low-grade sprains and chronic or remote stress changes are graded negative.

Medial meniscus tear: Abnormal signal that definitely contacts the meniscal surface on at least two images, or a morphologic abnormality such as a truncated, diminutive, or displaced fragment, involving the medial meniscus. Intrasubstance degeneration that does not reach the surface is negative.

Lateral meniscus tear: The same criteria applied to the lateral meniscus.

Medial compartment osteoarthritis: A moderate or large area (roughly 1 cm or greater) of high-grade cartilage loss, defined as greater than 50 percent of cartilage thickness, in the medial compartment, with or without underlying subchondral marrow changes.

Lateral compartment osteoarthritis: The same criteria applied to the lateral compartment.

Patellofemoral compartment osteoarthritis: The same criteria applied to the patellofemoral compartment.

Joint effusion: A moderate or large amount of fluid distending the joint.

Synovitis: Inflammation and thickening of the synovial lining of the joint.

Baker (popliteal) cyst: A moderate or large fluid collection in the characteristic location behind the knee.

Contusion: A bone contusion, seen as bone marrow edema-like signal from impact, without a discrete fracture line.

Acute fracture: An acute cortical break or fracture line.

Each study in the annotated reference set was independently labeled by two subspecialty-trained MSK radiologists, with disagreements adjudicated by a third radiologist to produce a single consensus ground truth. Labels are assigned at the level of the whole examination, for a single knee.

## Captions of the host's example figures

The same post shows 13 annotated MRI figures (yellow arrows and arrowheads mark the findings). Their captions, transcribed verbatim:

- Negative ACL tear: "Thickened ligament with increased signal and no discontinuity (arrows) and no secondary signs compatible with mucoid degeneration of the ACL"
- Positive ACL tear: "Complete ligament discontinuity in the intercondylar notch (arrows) with secondary signs including pivot shift bone contusions and deep notch sign (arrowheads)"
- Negative for MCL tear: "Intact fibers with ligament thickening (arrows) without edema signal adjacent to the ligament compatible with chronic stress changes or remote injury"
- Negative for MCL tear: "Intact fibers with increased signal at the femoral attachment without edema signal adjacent to the ligament (arrows) compatible with remote tear"
- Positive for MCL tear: "Torn fibers anteriorly with ligament thickening and edema signal within and adjacent to the ligament (arrowheads) with intact fibers posteriorly (arrows) compatible with an acute high-grade partial tear."
- Positive for MCL tear: "Intact fibers with ligament thickening and edema signal (arrows) and edema signal adjacent to the ligament (arrowheads) compatible with a high-grade partial tear"
- Negative for meniscal tear: "Increased signal definitely contacting surface on a single slice (arrow) with mild irregularity on adjacent slices compatible with fraying without a definite tear (arrowheads)"
- Negative for meniscal tear: "Increased signal (arrowheads) not definitely reaching surface compatible with degenerative changes. No tear."
- Positive for meniscal tear: "Increased signal definitely contacting surface on 2 slices (arrows) and parameniscal cyst (arrowhead) compatible with a meniscal tear"
- Negative OA: "Mild heterogenous signal (arrowheads) in the lateral tibial plateau due to normal orientation of cartilage fibers"
- Negative OA: "Normal homogeneous cartilage signal in the medial femoral condyle without underlying subchondral marrow changes"
- Positive OA: "Focal area of high-grade area of cartilage loss with subchondral reactive changes in the trochlea (arrowheads)."
- Positive OA: "Diffuse full thickness cartilage loss in medial femoral condyle with subchondral reactive changes (arrowheads)"
