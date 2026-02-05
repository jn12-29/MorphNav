# -*- coding: utf-8 -*-
import textwrap
import mujoco

BASE_XML = textwrap.dedent("""\
<mujoco model="biped_ant_style">
  <compiler angle="radian"/>

  <default>
    <motor ctrlrange="-1.0 1.0" ctrllimited="true" gear="75"/>
    <geom friction="1 0.5 0.5" solref=".02 1" solimp="0 .8 .01" material="self" density="50.0"/>
    <joint limited="true" armature="1" damping="1" stiffness="1" solreflimit=".04 1" solimplimit="0 .8 .03"/>
  </default>

  <asset>
    <material name="self" rgba=".8 .6 .4 1"/>
  </asset>

  <worldbody>
    <geom name="ground" type="plane" size="10 10 0.1" rgba=".9 .9 .9 1"/>
    <camera name="sideon" pos="0 -6 2.5" fovy="45" mode="targetbody" target="torso"/>
  </worldbody>

  <actuator>
    <motor name="left_hip"   joint="left_hip"/>
    <motor name="left_ankle" joint="left_ankle"/>
    <motor name="right_hip"   joint="right_hip"/>
    <motor name="right_ankle" joint="right_ankle"/>
  </actuator>

  <sensor>
    <touch name="torso_touch" site="torso_touch"/>
    <touch name="left_leg_touch" site="left_leg_touch"/>
    <touch name="left_ankle_touch" site="left_ankle_touch"/>
    <touch name="right_leg_touch" site="right_leg_touch"/>
    <touch name="right_ankle_touch" site="right_ankle_touch"/>

    <velocimeter  name="torso_vel"   site="torso_site"/>
    <gyro         name="torso_gyro"  site="torso_site"/>
    <accelerometer name="torso_accel" site="torso_site"/>
  </sensor>

  <contact>
    <exclude body1="torso" body2="left_aux"/>
    <exclude body1="torso" body2="right_aux"/>
  </contact>
</mujoco>
""")


def _add_free_base(body, name="root"):
    # 兼容不同版本：优先 add_freejoint，否则退化为 type=FREE 的 joint
    if hasattr(body, "add_freejoint"):
        body.add_freejoint(name=name)
    else:
        body.add_joint(name=name, type=mujoco.mjtJoint.mjJNT_FREE)


def add_ant_style_leg(torso, side: str, y_sign: float):
    """
    按 ant.xml 的“每条腿三层 body + 两个 hinge”的结构搭一条腿：
      <body name="{side}_leg">
        <geom name="{side}_aux_geom" .../>
        <body name="{side}_aux" pos="...">
          <joint name="{side}_hip" .../>
          <geom  name="{side}_leg_geom" .../>
          <site  name="{side}_leg_touch" .../>
          <body name="{side}_foot" pos="..." quat="...">
            <joint name="{side}_ankle" .../>
            <geom  name="{side}_ankle_geom" .../>
            <site  name="{side}_ankle_touch" .../>
          </body>
        </body>
      </body>
    """

    # ------- 尺寸参数 -------
    hip_y = 0.18 * y_sign         # 左右髋位置
    hip_z = -0.05                 # 略微向下
    thigh_len = 0.35
    shin_len  = 0.35
    r = 0.08                      # capsule 半径
    touch_box = [0.10, 0.10, 0.25]

    # ant 里 foot 用 quat 做了一个固定倾角
    if side == "left":
        foot_quat = [0.98480775301220802, -0.33428158105097677, 0.33428158105097677, 0.0]
        ankle_axis = [1, -1, 0]   # ant "斜轴”
    else:
        foot_quat = [0.98480775301220802,  0.33428158105097677, 0.33428158105097677, 0.0]
        ankle_axis = [-1, -1, 0]

    # ------- 外层 leg body-------
    leg = torso.add_body(name=f"{side}_leg")

    # aux_geom：从 torso 指向髋的位置
    leg.add_geom(
        name=f"{side}_aux_geom",
        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        size=[r, r, r],
        fromto=[0.0, 0.0, 0.0, 0.0, hip_y, hip_z],
    )

    # ------- aux body（ hip joint ） -------
    aux = leg.add_body(name=f"{side}_aux", pos=[0.0, hip_y, hip_z])

    # “两足站立”，这里用 pitch（绕 y 轴）；
    aux.add_joint(
        name=f"{side}_hip",
        type=mujoco.mjtJoint.mjJNT_HINGE,
        axis=[0, 1, 0],
        range=[-0.7, 0.7],
    )

    # 上腿 capsule：从髋往下
    aux.add_geom(
        name=f"{side}_leg_geom",
        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        size=[r, r, r],
        fromto=[0.0, 0.0, 0.0, 0.0, 0.0, -thigh_len],
    )

    # 上腿 touch site
    aux.add_site(
        name=f"{side}_leg_touch",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0.0, 0.0, -thigh_len * 0.5],
        size=touch_box,
        zaxis=[0, 0, -1],
        rgba=[1, 1, 0, 0],
        group=4,
    )

    # ------- foot body（放 ankle joint 的那层） -------
    foot = aux.add_body(
        name=f"{side}_foot",
        pos=[0.0, 0.0, -thigh_len],
        quat=foot_quat,
    )

    foot.add_joint(
        name=f"{side}_ankle",
        type=mujoco.mjtJoint.mjJNT_HINGE,
        axis=ankle_axis,
        range=[-0.5, 0.5],
    )

    # 小腿 capsule：从膝/踝往下（这里直接在 foot body 上画第二段）
    foot.add_geom(
        name=f"{side}_ankle_geom",
        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        size=[r, r, r],
        fromto=[0.0, 0.0, 0.0, 0.0, 0.0, -shin_len],
    )

    foot.add_site(
        name=f"{side}_ankle_touch",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0.0, 0.0, -shin_len * 0.5],
        size=[0.10, 0.10, 0.30],
        zaxis=[0, 0, -1],
        rgba=[1, 1, 0, 0],
        group=4,
    )


def build_biped_ant_style_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_string(BASE_XML)

    # -------------- torso body --------------
    torso = spec.worldbody.add_body(name="torso", pos=[0, 0, 0.90])
    _add_free_base(torso, name="root")

    torso.add_geom(
        name="torso_geom",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.20,0.20,0.20],
        density=100.0,
    )

    torso.add_site(
        name="torso_touch",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.22, 0.16, 0.18],
        rgba=[0, 0, 1, 1],
        group=4,
    )
    torso.add_site(
        name="torso_site",
        size=[0.05, 0.05, 0.05],
        rgba=[1, 0, 0, 1],
    )

    # 两条腿：left/right
    add_ant_style_leg(torso, side="left",  y_sign=+1.0)
    add_ant_style_leg(torso, side="right", y_sign=-1.0)

    return spec


if __name__ == "__main__":
    spec = build_biped_ant_style_spec()
    spec.compile()
    xml = spec.to_xml()

    with open("biped_ant_style.xml", "w", encoding="utf-8") as f:
        f.write(xml)

    print("[OK] wrote biped_ant_style.xml")
