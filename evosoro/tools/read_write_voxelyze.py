import hashlib
import os
import time
import random
import re
import numpy as np

def read_voxlyze_results(population, print_log, filename="softbotsOutput.xml"):
    i = 0
    max_attempts = 60
    file_size = 0
    this_file = ""
    re_float_array = re.compile("( [-0-9.e+]+)")

    while (i < max_attempts) and (file_size == 0):
        try:
            file_size = os.stat(filename).st_size
            this_file = open(filename)
        except ImportError:  # TODO: is this the correct exception?
            file_size = 0
        i += 1
        time.sleep(1)

    if file_size == 0:
        print_log.message("ERROR: Cannot find a non-empty fitness file in %d attempts: abort" % max_attempts)
        exit(1)

    results = {rank: None for rank in range(len(population.objective_dict))}
    this_file = open(filename)  # TODO: is there a way to just go back to the first line without reopening the file?
    fContent = str(this_file.read())
    for rank, details in population.objective_dict.items():
        tag = details["tag"]
#        isArray = details["isArray"]
#        evalFun = details["evalFun"]

        if tag is not None:
            re_match_tag_arg = re.compile('<{tag}>.+</{tag}>'.format(tag = tag[1:-1]))
            matches = re_match_tag_arg.findall(fContent)
            more_values_same_tag = list()
            for n_match in matches:
                content = n_match[n_match.find(tag) + len(tag):n_match.find("</" + tag[1:])]
                if re_float_array.match(content) is None:
                    element = abs(float(content))
                else:
                    iter = re_float_array.finditer(content)
                    element = list()
                    for el in iter:
                        if el:
                            element.append(float(el.group(1)))
                if len(matches) > 1:
                    more_values_same_tag.append(element)
                else:
                    more_values_same_tag = element

            results[rank] = more_values_same_tag
#            if not isArray:
#               for line in this_file:
#                    if tag in line:
#                        results[rank] = abs(float(line[line.find(tag) + len(tag):line.find("</" + tag[1:])]))
#            else:
#                res = []
#                for line in this_file:
#                    if tag in line:
#                        res.extend([abs(float(line[line.find(tag) + len(tag):line.find("</" + tag[1:])]))])
#                results[rank] = evalFun(res)
#            for line in this_file:
#                if tag[1:-1] in line:
#                if '<N'+tag[1:-1]+'>' in line:
#                        Nstr = line[line.find(tag) + len(tag):line.find("</" + tag[1:])]
#                        Nrep = abs(float(Nstr))
#                if tag in line:
#                str = line[line.find(tag) + len(tag):line.find("</" + tag[1:])]
#                if re_float_array.match(str) is None:
#                    results[rank] = abs(float(str))
#                else:
#                    iter = re_float_array.finditer(str)
#                    elements = list()
#                    for el in iter:
#                       if el:
#                            elements.append(float(el.group(1)))
#                    results[rank] = elements
    return results


def write_voxelyze_file(sim, env, individual, run_directory, run_name):

    # TODO: work in base.py to remove redundant static text in this function

    # update any env variables based on outputs instead of writing outputs in
    for name, details in individual.genotype.to_phenotype_mapping.items():
        if details["env_kws"] is not None:
            for env_key, env_func in details["env_kws"].items():
                setattr(env, env_key, env_func(details["state"]))  # currently only used when evolving frequency
                # print env_key, env_func(details["state"])

    voxelyze_file = open(run_directory + "/voxelyzeFiles/" + run_name + "--id_%05i.vxa" % individual.id, "w")

    voxelyze_file.write(
        "<?xml version=\"1.0\" encoding=\"ISO-8859-1\"?>\n\
        <VXA Version=\"1.0\">\n\
        <Simulator>\n")

    # Sim
    for name, tag in sim.new_param_tag_dict.items():
        voxelyze_file.write(tag + str(getattr(sim, name)) + "</" + tag[1:] + "\n")

    voxelyze_file.write(
        "<Integration>\n\
        <Integrator>0</Integrator>\n\
        <DtFrac>" + str(sim.dt_frac) + "</DtFrac>\n\
        </Integration>\n\
        <Damping>\n\
        <BondDampingZ>1</BondDampingZ>\n\
        <ColDampingZ>0.8</ColDampingZ>\n\
        <SlowDampingZ>0.01</SlowDampingZ>\n\
        </Damping>\n\
        <Collisions>\n\
        <SelfColEnabled>" + str(int(sim.self_collisions_enabled)) + "</SelfColEnabled>\n\
        <ColSystem>3</ColSystem>\n\
        <CollisionHorizon>2</CollisionHorizon>\n\
        </Collisions>\n\
        <Features>\n\
        <FluidDampEnabled>0</FluidDampEnabled>\n\
        <PoissonKickBackEnabled>0</PoissonKickBackEnabled>\n\
        <EnforceLatticeEnabled>0</EnforceLatticeEnabled>\n\
        </Features>\n\
        <SurfMesh>\n\
        <CMesh>\n\
        <DrawSmooth>1</DrawSmooth>\n\
        <Vertices/>\n\
        <Facets/>\n\
        <Lines/>\n\
        </CMesh>\n\
        </SurfMesh>\n\
        <StopCondition>\n\
        <StopConditionType>" + str(int(sim.stop_condition)) + "</StopConditionType>\n\
        <StopConditionValue>" + str(sim.simulation_time) + "</StopConditionValue>\n\
        <InitCmTime>" + str(sim.fitness_eval_init_time) + "</InitCmTime>\n\
        </StopCondition>\n\
        <EquilibriumMode>\n\
        <EquilibriumModeEnabled>" + str(sim.equilibrium_mode) + "</EquilibriumModeEnabled>\n\
        </EquilibriumMode>\n\
        <GA>\n\
        <WriteFitnessFile>1</WriteFitnessFile>\n\
        <FitnessFileName>" + run_directory + "/fitnessFiles/softbotsOutput--id_%05i.xml" % individual.id +
        "</FitnessFileName>\n\
        <QhullTmpFile>" + run_directory + "/tempFiles/qhullInput--id_%05i.txt" % individual.id + "</QhullTmpFile>\n\
        <CurvaturesTmpFile>" + run_directory + "/tempFiles/curvatures--id_%05i.txt" % individual.id +
        "</CurvaturesTmpFile>\n\
        </GA>\n\
        <MinTempFact>" + str(sim.min_temp_fact) + "</MinTempFact>\n\
        <MaxTempFactChange>" + str(sim.max_temp_fact_change) + "</MaxTempFactChange>\n\
        <MaxStiffnessChange>" + str(sim.max_stiffness_change) + "</MaxStiffnessChange>\n\
        <MinElasticMod>" + str(sim.min_elastic_mod) + "</MinElasticMod>\n\
        <MaxElasticMod>" + str(sim.max_elastic_mod) + "</MaxElasticMod>\n\
        <ErrorThreshold>" + str(0) + "</ErrorThreshold>\n\
        <ThresholdTime>" + str(0) + "</ThresholdTime>\n\
        <MaxKP>" + str(0) + "</MaxKP>\n\
        <MaxKI>" + str(0) + "</MaxKI>\n\
        <MaxANTIWINDUP>" + str(0) + "</MaxANTIWINDUP>\n\
        </Simulator>\n")

    # Env
    voxelyze_file.write(
        "<Environment>\n")
    for name, tag in env.new_param_tag_dict.items():
        voxelyze_file.write(tag + str(getattr(env, name)) + "</" + tag[1:] + "\n")

    voxelyze_file.write(
        "<Boundary_Conditions>\n\
          <NumBCs>4</NumBCs>\n\
          <FRegion>\n\
            <PrimType>0</PrimType>\n\
            <X>0</X>\n\
            <Y>0</Y>\n\
            <Z>0</Z>\n\
            <dX>0.01</dX>\n\
            <dY>1</dY>\n\
            <dZ>1</dZ>\n\
            <Radius>0</Radius>\n\
            <R>0.4</R>\n\
            <G>0.6</G>\n\
            <B>0.4</B>\n\
            <alpha>1</alpha>\n\
            <DofFixed>63</DofFixed>\n\
            <ForceX>0</ForceX>\n\
            <ForceY>0</ForceY>\n\
            <ForceZ>0</ForceZ>\n\
            <TorqueX>0</TorqueX>\n\
            <TorqueY>0</TorqueY>\n\
            <TorqueZ>0</TorqueZ>\n\
            <DisplaceX>0</DisplaceX>\n\
            <DisplaceY>0</DisplaceY>\n\
            <DisplaceZ>0</DisplaceZ>\n\
            <AngDisplaceX>0</AngDisplaceX>\n\
            <AngDisplaceY>0</AngDisplaceY>\n\
            <AngDisplaceZ>0</AngDisplaceZ>\n\
          </FRegion>\n\
          <FRegion>\n\
            <PrimType>0</PrimType>\n\
            <X>0</X>\n\
            <Y>0</Y>\n\
            <Z>0</Z>\n\
            <dX>1</dX>\n\
            <dY>0.01</dY>\n\
            <dZ>1</dZ>\n\
            <Radius>0</Radius>\n\
            <R>0.4</R>\n\
            <G>0.6</G>\n\
            <B>0.4</B>\n\
            <alpha>1</alpha>\n\
            <DofFixed>63</DofFixed>\n\
            <ForceX>0</ForceX>\n\
            <ForceY>0</ForceY>\n\
            <ForceZ>0</ForceZ>\n\
            <TorqueX>0</TorqueX>\n\
            <TorqueY>0</TorqueY>\n\
            <TorqueZ>0</TorqueZ>\n\
            <DisplaceX>0</DisplaceX>\n\
            <DisplaceY>0</DisplaceY>\n\
            <DisplaceZ>0</DisplaceZ>\n\
            <AngDisplaceX>0</AngDisplaceX>\n\
            <AngDisplaceY>0</AngDisplaceY>\n\
            <AngDisplaceZ>0</AngDisplaceZ>\n\
          </FRegion>\n\
          <FRegion>\n\
            <PrimType>0</PrimType>\n\
            <X>0.99</X>\n\
            <Y>0</Y>\n\
            <Z>0</Z>\n\
            <dX>0.01</dX>\n\
            <dY>1</dY>\n\
            <dZ>1</dZ>\n\
            <Radius>0</Radius>\n\
            <R>0.4</R>\n\
            <G>0.6</G>\n\
            <B>0.4</B>\n\
            <alpha>1</alpha>\n\
            <DofFixed>63</DofFixed>\n\
            <ForceX>0</ForceX>\n\
            <ForceY>0</ForceY>\n\
            <ForceZ>0</ForceZ>\n\
            <TorqueX>0</TorqueX>\n\
            <TorqueY>0</TorqueY>\n\
            <TorqueZ>0</TorqueZ>\n\
            <DisplaceX>0</DisplaceX>\n\
            <DisplaceY>0</DisplaceY>\n\
            <DisplaceZ>0</DisplaceZ>\n\
            <AngDisplaceX>0</AngDisplaceX>\n\
            <AngDisplaceY>0</AngDisplaceY>\n\
            <AngDisplaceZ>0</AngDisplaceZ>\n\
          </FRegion>\n\
          <FRegion>\n\
            <PrimType>0</PrimType>\n\
            <X>0</X>\n\
            <Y>0.99</Y>\n\
            <Z>0</Z>\n\
            <dX>1</dX>\n\
            <dY>0.01</dY>\n\
            <dZ>1</dZ>\n\
            <Radius>0</Radius>\n\
            <R>0</R>\n\
            <G>1</G>\n\
            <B>0</B>\n\
            <alpha>1</alpha>\n\
            <DofFixed>63</DofFixed>\n\
            <ForceX>0</ForceX>\n\
            <ForceY>0</ForceY>\n\
            <ForceZ>0</ForceZ>\n\
            <TorqueX>0</TorqueX>\n\
            <TorqueY>0</TorqueY>\n\
            <TorqueZ>0</TorqueZ>\n\
            <DisplaceX>0</DisplaceX>\n\
            <DisplaceY>0</DisplaceY>\n\
            <DisplaceZ>0</DisplaceZ>\n\
            <AngDisplaceX>0</AngDisplaceX>\n\
            <AngDisplaceY>0</AngDisplaceY>\n\
            <AngDisplaceZ>0</AngDisplaceZ>\n\
          </FRegion>\n\
        </Boundary_Conditions>\n\
        <Gravity>\n\
        <GravEnabled>" + str(env.gravity_enabled) + "</GravEnabled>\n\
        <GravAcc>-9.81</GravAcc>\n\
        <FloorEnabled>" + str(env.floor_enabled) + "</FloorEnabled>\n\
        <FloorSlope>" + str(env.floor_slope) + "</FloorSlope>\n\
        </Gravity>\n\
        <Thermal>\n\
        <TempEnabled>" + str(env.temp_enabled) + "</TempEnabled>\n\
        <TempAmp>39</TempAmp>\n\
        <TempBase>25</TempBase>\n\
        <VaryTempEnabled>1</VaryTempEnabled>\n\
        <TempPeriod>" + str(1.0 / env.frequency) + "</TempPeriod>\n\
        </Thermal>\n\
        <TimeBetweenTraces>" + str(env.time_between_traces) + "</TimeBetweenTraces>\n\
        <StickyFloor>" + str(env.sticky_floor) + "</StickyFloor>\n\
        </Environment>\n")

    voxelyze_file.write(
        "<VXC Version=\"0.93\">\n\
        <Lattice>\n\
        <Lattice_Dim>" + str(env.lattice_dimension) + "</Lattice_Dim>\n\
        <X_Dim_Adj>1</X_Dim_Adj>\n\
        <Y_Dim_Adj>1</Y_Dim_Adj>\n\
        <Z_Dim_Adj>1</Z_Dim_Adj>\n\
        <X_Line_Offset>0</X_Line_Offset>\n\
        <Y_Line_Offset>0</Y_Line_Offset>\n\
        <X_Layer_Offset>0</X_Layer_Offset>\n\
        <Y_Layer_Offset>0</Y_Layer_Offset>\n\
        </Lattice>\n\
        <Voxel>\n\
        <Vox_Name>BOX</Vox_Name>\n\
        <X_Squeeze>1</X_Squeeze>\n\
        <Y_Squeeze>1</Y_Squeeze>\n\
        <Z_Squeeze>1</Z_Squeeze>\n\
        </Voxel>\n\
        <Palette>\n\
        <Material ID=\"1\">\n\
        <MatType>0</MatType>\n\
        <Name>Passive_Soft</Name>\n\
        <Display>\n\
        <Red>0</Red>\n\
        <Green>1</Green>\n\
        <Blue>1</Blue>\n\
        <Alpha>1</Alpha>\n\
        </Display>\n\
        <Mechanical>\n\
        <MatModel>0</MatModel>\n\
        <Elastic_Mod>" + str(env.softest_material) + "e+007</Elastic_Mod>\n\
        <Plastic_Mod>0</Plastic_Mod>\n\
        <Yield_Stress>0</Yield_Stress>\n\
        <FailModel>0</FailModel>\n\
        <Fail_Stress>0</Fail_Stress>\n\
        <Fail_Strain>0</Fail_Strain>\n\
        <Density>1e+006</Density>\n\
        <Poissons_Ratio>0.35</Poissons_Ratio>\n\
        <CTE>0</CTE>\n\
        <uStatic>1</uStatic>\n\
        <uDynamic>0.5</uDynamic>\n\
        </Mechanical>\n\
        </Material>\n\
        <Material ID=\"2\">\n\
        <MatType>0</MatType>\n\
        <Name>Passive_Hard</Name>\n\
        <Display>\n\
        <Red>0</Red>\n\
        <Green>0</Green>\n\
        <Blue>1</Blue>\n\
        <Alpha>1</Alpha>\n\
        </Display>\n\
        <Mechanical>\n\
        <MatModel>0</MatModel>\n\
        <Elastic_Mod>" + str(env.softest_material * 2) + "e+008</Elastic_Mod>\n\
        <Plastic_Mod>0</Plastic_Mod>\n\
        <Yield_Stress>0</Yield_Stress>\n\
        <FailModel>0</FailModel>\n\
        <Fail_Stress>0</Fail_Stress>\n\
        <Fail_Strain>0</Fail_Strain>\n\
        <Density>1e+006</Density>\n\
        <Poissons_Ratio>0.35</Poissons_Ratio>\n\
        <CTE>0</CTE>\n\
        <uStatic>1</uStatic>\n\
        <uDynamic>0.5</uDynamic>\n\
        </Mechanical>\n\
        </Material>\n\
        <Material ID=\"3\">\n\
        <MatType>0</MatType>\n\
        <Name>Active_+</Name>\n\
        <Display>\n\
        <Red>1</Red>\n\
        <Green>0</Green>\n\
        <Blue>0</Blue>\n\
        <Alpha>1</Alpha>\n\
        </Display>\n\
        <Mechanical>\n\
        <MatModel>0</MatModel>\n\
        <Elastic_Mod>" + str(env.material_stiffness) + "</Elastic_Mod>\n\
        <Plastic_Mod>0</Plastic_Mod>\n\
        <Yield_Stress>0</Yield_Stress>\n\
        <FailModel>0</FailModel>\n\
        <Fail_Stress>0</Fail_Stress>\n\
        <Fail_Strain>0</Fail_Strain>\n\
        <Density>1e+006</Density>\n\
        <Poissons_Ratio>0.35</Poissons_Ratio>\n\
        <CTE>" + str(0.01*(1+random.uniform(0, env.actuation_variance))) + "</CTE>\n\
        <uStatic>1</uStatic>\n\
        <uDynamic>0.5</uDynamic>\n\
        </Mechanical>\n\
        </Material>\n\
        <Material ID=\"4\">\n\
        <MatType>0</MatType>\n\
        <Name>Active_-</Name>\n\
        <Display>\n\
        <Red>0</Red>\n\
        <Green>1</Green>\n\
        <Blue>0</Blue>\n\
        <Alpha>1</Alpha>\n\
        </Display>\n\
        <Mechanical>\n\
        <MatModel>0</MatModel>\n\
        <Elastic_Mod>" + str(env.softest_material) + "e+006</Elastic_Mod>\n\
        <Plastic_Mod>0</Plastic_Mod>\n\
        <Yield_Stress>0</Yield_Stress>\n\
        <FailModel>0</FailModel>\n\
        <Fail_Stress>0</Fail_Stress>\n\
        <Fail_Strain>0</Fail_Strain>\n\
        <Density>1e+006</Density>\n\
        <Poissons_Ratio>0.35</Poissons_Ratio>\n\
        <CTE>" + str(-0.01*(1+random.uniform(0, env.actuation_variance))) + "</CTE>\n\
        <uStatic>1</uStatic>\n\
        <uDynamic>0.5</uDynamic>\n\
        </Mechanical>\n\
        </Material>\n\
        </Palette>\n\
        <Structure Compression=\"ASCII_READABLE\">\n\
        <X_Voxels>" + str(individual.genotype.orig_size_xyz[0]) + "</X_Voxels>\n\
        <Y_Voxels>" + str(individual.genotype.orig_size_xyz[1]) + "</Y_Voxels>\n\
        <Z_Voxels>21</Z_Voxels>\n")

    all_tags = [details["tag"] for name, details in individual.genotype.to_phenotype_mapping.items()]
    if "<Data>" not in all_tags:  # not evolving topology -- fixed presence/absence of voxels
        voxelyze_file.write("<Data>\n")
        for z in range(21):
            voxelyze_file.write("<Layer><![CDATA[")
            for y in range(individual.genotype.orig_size_xyz[1]):
                for x in range(individual.genotype.orig_size_xyz[0]):
                    if z == 0:
                        voxelyze_file.write(str(env.fixed_shape[z][y][x]))
                    else:
                        voxelyze_file.write(str(0))

            voxelyze_file.write("]]></Layer>\n")
        voxelyze_file.write("</Data>\n")

    # append custom parameters
    string_for_md5 = ""

    for name, details in individual.genotype.to_phenotype_mapping.items():

        # start tag
        voxelyze_file.write(details["tag"]+"\n")

        # record any additional params associated with the output
        if details["params"] is not None:
            for param_tag, param in zip(details["param_tags"], details["params"]):
                voxelyze_file.write(param_tag + str(param) + "</" + param_tag[1:] + "\n")

        if details["env_kws"] is None:
            # write the output state matrix to file
            for z in range(individual.genotype.orig_size_xyz[2]):
                voxelyze_file.write("<Layer><![CDATA[")
                for y in range(individual.genotype.orig_size_xyz[1]):
                    for x in range(individual.genotype.orig_size_xyz[0]):

                        state = details["output_type"](details["state"][x, y, z])
                        # for n, network in enumerate(individual.genotype):
                        #     if name in network.output_node_names:
                        #         state = individual.genotype[n].graph.node[name]["state"][x, y, z]

                        voxelyze_file.write(str(state))
                        if details["tag"] != "<Data>":  # TODO more dynamic
                            voxelyze_file.write(", ")
                        string_for_md5 += str(state)

                voxelyze_file.write("]]></Layer>\n")

        # end tag
        voxelyze_file.write("</" + details["tag"][1:] + "\n")


    voxelyze_file.write(
        "</Structure>\n\
        <Scenarios>\n")

    if env.scenarios is not None:
        write_scenarios(env, individual, voxelyze_file)

    voxelyze_file.write(
        "</Scenarios>\n\
        </VXC>\n\
        </VXA>")
    voxelyze_file.close()

    m = hashlib.md5()
    m.update(string_for_md5)

    return m.hexdigest()

def write_scenarios(env, individual, voxelyze_file):

    voxelyze_file.write("<NScenarios>" + str(len(env.scenarios)) + "</NScenarios>\n")

    voxelyze_file.write("<ScenarioNames>\n")

    for name, scenario in env.scenarios.iteritems():
        voxelyze_file.write("<ScenarioName>" + name + "</ScenarioName>\n")

    voxelyze_file.write("</ScenarioNames>\n")

    voxelyze_file.write("<ScenarioShapes>\n")

    for name, scenario in env.scenarios.iteritems():
        voxelyze_file.write("<ScenarioData>\n")
        for z in range(21):
            voxelyze_file.write("<Layer><![CDATA[")
            for y in range(individual.genotype.orig_size_xyz[1]):
                for x in range(individual.genotype.orig_size_xyz[0]):
                    voxelyze_file.write(str(scenario[z][y][x]))
            voxelyze_file.write("]]></Layer>\n")
        voxelyze_file.write("</ScenarioData>\n")

    voxelyze_file.write("</ScenarioShapes>\n")
